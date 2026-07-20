"""Compact PyTorch DDSP synthesizer — architecture + differentiable synthesis.

This is the neural sarangi *timbre* model. It replaces Magenta's (archived,
TensorFlow-only, arm64-hostile) DDSP with a small, self-contained PyTorch
harmonic-plus-filtered-noise synth in the spirit of the DDSP paper
(Engel et al., 2020) and the acids-ircam/ddsp_pytorch reference.

Why it lives in ``sarangi_gen`` (the shipped package): the runtime ``ddsp``
backend needs the *same* module to rebuild the graph and load weights. It is
imported **lazily** (only when the ddsp backend actually synthesizes), so plain
``import sarangi_gen`` never pulls in torch — torch is an optional, backend-only
dependency, exactly as tensorflow/ddsp were before.

Control signals run at a low frame rate (``sampling_rate / block_size``); the
model emits per-frame harmonic amplitudes + noise-band gains that are upsampled
to audio rate and turned into a waveform by two differentiable synths:

  * harmonic oscillator — additive sines on a continuous, f0-integrated phase
    (so pitch changes/meend are C0-continuous, like the sine backend);
  * filtered noise — white noise shaped by a per-frame FIR built from the
    predicted band gains (the sarangi's bow noise / breathiness).

The same forward pass is used for training (audio reconstruction loss) and for
runtime inference (drive it with the render contour's f0 + loudness).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Config (persisted next to the weights so the graph is rebuildable at load)
# --------------------------------------------------------------------------- #
CKPT_WEIGHTS = "ddsp_torch.pt"      # model + optimizer + step (training state)
CKPT_CONFIG = "config.json"         # DDSPConfig, for rebuilding the graph


@dataclass
class DDSPConfig:
    sampling_rate: int = 16000
    block_size: int = 160            # hop; frame rate = sr / block_size = 100 Hz
    hidden_size: int = 256
    n_harmonic: int = 100
    n_bands: int = 65                # filtered-noise frequency bands
    gru_layers: int = 1
    # Harmonic bandlimit: amplitudes are held full below ``rolloff_lo_hz`` and
    # cosine-tapered to zero by ``rolloff_hi_hz`` (well under Nyquist). This
    # kills the near-Nyquist "whistle/feedback" artifact the additive synth
    # otherwise produces at low f0 (where many harmonics fit under Nyquist).
    rolloff_lo_hz: float = 5000.0
    rolloff_hi_hz: float = 7000.0
    # Loudness feature is A-weighted dB min-max normalized to [0,1] over the
    # dataset using these percentiles; stored so training/inference agree.
    loudness_db_lo: float = -80.0
    loudness_db_hi: float = 0.0

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DDSPConfig":
        with open(path) as f:
            return cls(**json.load(f))


# --------------------------------------------------------------------------- #
# Differentiable synthesis primitives
# --------------------------------------------------------------------------- #
def modified_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """DDSP-paper output nonlinearity: strictly positive, saturating at ~2."""
    return 2.0 * torch.sigmoid(x) ** math.log(10.0) + 1e-7


def upsample(signal: torch.Tensor, factor: int) -> torch.Tensor:
    """Linear-interpolate a frame-rate control signal [B, T, C] to [B, T*factor, C]."""
    signal = signal.permute(0, 2, 1)
    signal = F.interpolate(signal, size=signal.shape[-1] * factor,
                           mode="linear", align_corners=False)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes: torch.Tensor, pitch: torch.Tensor,
                         sampling_rate: int) -> torch.Tensor:
    """Zero out harmonics whose frequency would alias (>= Nyquist)."""
    n_harm = amplitudes.shape[-1]
    k = torch.arange(1, n_harm + 1, device=pitch.device)
    freqs = pitch * k                                   # [B, T, n_harm]
    mask = (freqs < sampling_rate / 2).float() + 1e-7
    return amplitudes * mask


def harmonic_rolloff(amplitudes: torch.Tensor, pitch: torch.Tensor,
                     lo_hz: float, hi_hz: float) -> torch.Tensor:
    """Cosine-taper harmonic amplitudes to zero between ``lo_hz`` and ``hi_hz``.

    Weight is 1 for harmonics below ``lo_hz`` and 0 above ``hi_hz`` (a raised
    cosine in between), bandlimiting the additive synth so it cannot emit a loud
    near-Nyquist whistle. Applied in both training and inference so the model
    learns its timbre within the band.
    """
    n_harm = amplitudes.shape[-1]
    k = torch.arange(1, n_harm + 1, device=pitch.device)
    freqs = pitch * k                                   # [B, T, n_harm]
    frac = ((freqs - lo_hz) / max(hi_hz - lo_hz, 1e-6)).clamp(0.0, 1.0)
    weight = 0.5 * (1.0 + torch.cos(math.pi * frac))    # 1 -> 0
    return amplitudes * weight


def harmonic_synth(pitch: torch.Tensor, amplitudes: torch.Tensor,
                   sampling_rate: int) -> torch.Tensor:
    """Additive sine bank on a shared, f0-integrated phase (audio rate).

    ``pitch`` [B, N, 1] and ``amplitudes`` [B, N, n_harmonic] are already
    upsampled to the audio rate. Phase is the cumulative integral of the
    instantaneous frequency, so a pitch change only bends the phase slope —
    the waveform stays continuous across meend glides and note boundaries.
    """
    n_harm = amplitudes.shape[-1]
    k = torch.arange(1, n_harm + 1, device=pitch.device)
    omega = torch.cumsum(2.0 * math.pi * pitch / sampling_rate, dim=1)  # [B,N,1]
    phases = omega * k                                                  # [B,N,n_harm]
    return (torch.sin(phases) * amplitudes).sum(-1, keepdim=True)       # [B,N,1]


def amp_to_impulse_response(amp: torch.Tensor, target_size: int) -> torch.Tensor:
    """Turn per-frame band magnitudes into a zero-phase, windowed FIR."""
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)                          # [B, T, filter_size]
    filter_size = amp.shape[-1]
    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)
    amp = amp * win
    amp = F.pad(amp, (0, int(target_size) - filter_size))
    return torch.roll(amp, -filter_size // 2, -1)


def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Causal FFT convolution of a per-frame signal block with its FIR."""
    signal = F.pad(signal, (0, signal.shape[-1]))
    kernel = F.pad(kernel, (kernel.shape[-1], 0))
    out = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    return out[..., out.shape[-1] // 2:]


def _mlp(in_size: int, hidden: int, n_layers: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n_layers):
        layers += [nn.Linear(in_size if i == 0 else hidden, hidden),
                   nn.LayerNorm(hidden), nn.LeakyReLU()]
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class DDSPSynth(nn.Module):
    """Decoder-only DDSP: (f0, loudness) frames -> waveform.

    Self-supervised: at train time f0/loudness are extracted from the target
    audio and the model learns to reconstruct it (multi-scale spectral loss).
    """

    def __init__(self, cfg: DDSPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        self.in_pitch = _mlp(1, h)
        self.in_loud = _mlp(1, h)
        self.gru = nn.GRU(2 * h, h, num_layers=cfg.gru_layers, batch_first=True)
        self.out_mlp = _mlp(h + 2 * h, h)
        # +1 for the global harmonic amplitude, then the harmonic distribution;
        # a second head for the noise band gains.
        self.proj_harm = nn.Linear(h, cfg.n_harmonic + 1)
        self.proj_noise = nn.Linear(h, cfg.n_bands)

    def forward(self, pitch: torch.Tensor, loudness: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """pitch [B,T,1] in Hz, loudness [B,T,1] in [0,1]  ->  audio [B, T*block].

        With ``return_components=True`` returns ``(harmonic, noise)`` separately
        (same shape) — used for diagnostics, not the render path.
        """
        cfg = self.cfg
        hidden = torch.cat([self.in_pitch(pitch), self.in_loud(loudness)], -1)
        gru_out, _ = self.gru(hidden)
        hidden = self.out_mlp(torch.cat([gru_out, hidden], -1))

        # --- harmonic branch ---
        harm = self.proj_harm(hidden)
        total_amp = modified_sigmoid(harm[..., :1])
        distribution = modified_sigmoid(harm[..., 1:])
        distribution = distribution / distribution.sum(-1, keepdim=True)
        amplitudes = distribution * total_amp
        amplitudes = remove_above_nyquist(amplitudes, pitch, cfg.sampling_rate)
        amplitudes = harmonic_rolloff(amplitudes, pitch,
                                      cfg.rolloff_lo_hz, cfg.rolloff_hi_hz)
        amplitudes = upsample(amplitudes, cfg.block_size)
        pitch_up = upsample(pitch, cfg.block_size)
        harmonic = harmonic_synth(pitch_up, amplitudes, cfg.sampling_rate)  # [B,N,1]

        # --- filtered-noise branch ---
        noise_gains = modified_sigmoid(self.proj_noise(hidden) - 5.0)  # start quiet
        ir = amp_to_impulse_response(noise_gains, cfg.block_size)       # [B,T,block]
        white = (torch.rand(ir.shape[0], ir.shape[1], cfg.block_size,
                            device=ir.device) * 2.0 - 1.0)
        noise = fft_convolve(white, ir).reshape(ir.shape[0], -1, 1)     # [B,N,1]

        if return_components:
            return harmonic.squeeze(-1), noise.squeeze(-1)
        signal = harmonic + noise
        return signal.squeeze(-1)                                       # [B, N]


# --------------------------------------------------------------------------- #
# Load helper (used by the runtime backend)
# --------------------------------------------------------------------------- #
def load_synth(checkpoint_dir: str, device: str = "cpu") -> tuple[DDSPSynth, DDSPConfig]:
    """Rebuild the model from ``config.json`` and restore weights."""
    cfg = DDSPConfig.from_json(os.path.join(checkpoint_dir, CKPT_CONFIG))
    model = DDSPSynth(cfg).to(device)
    state = torch.load(os.path.join(checkpoint_dir, CKPT_WEIGHTS), map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg
