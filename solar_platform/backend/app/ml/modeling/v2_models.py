"""
app.ml.modeling.v2_models
-------------------------
V2 Physics-Informed Multi-Head Neural Architecture for Solar Cell Surrogate Learning.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class ModuleDummy:
        pass
    nn = ModuleDummy()
    nn.Module = object


class FourierCoordinateEncoder(nn.Module if TORCH_AVAILABLE else object):
    """Projects 1D scalar coordinate x into K high-frequency sine/cosine embeddings."""

    def __init__(self, num_frequencies: int = 8, scale: float = 1.0):
        if TORCH_AVAILABLE:
            super().__init__()
        self.num_frequencies = num_frequencies
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)

        freq_bands = 2.0 ** torch.arange(self.num_frequencies, device=x.device, dtype=x.dtype) * math.pi * self.scale
        x_freq = x * freq_bands.unsqueeze(0)
        sin_feat = torch.sin(x_freq)
        cos_feat = torch.cos(x_freq)
        return torch.cat([sin_feat, cos_feat], dim=-1)


class TargetNormalizer:
    """Manages target Z-Score normalization per scientific type."""

    def __init__(self):
        self.means: Dict[str, float] = {"JV": 12.0, "QE": 45.0, "RESPONSIVITY": 0.2}
        self.stds: Dict[str, float] = {"JV": 10.0, "QE": 30.0, "RESPONSIVITY": 0.25}

    def fit(self, targets_by_type: Dict[str, List[float]]):
        """Fit target mean and std per scientific type."""
        for stype, vals in targets_by_type.items():
            if len(vals) > 1:
                arr = np.array(vals, dtype=np.float64)
                m = float(np.mean(arr))
                s = float(np.std(arr))
                self.means[stype] = m
                self.stds[stype] = s if s > 1e-6 else 1.0

    def normalize(self, y: float, stype: str) -> float:
        m = self.means.get(stype, 0.0)
        s = self.stds.get(stype, 1.0)
        return (y - m) / s

    def denormalize(self, y_norm: float, stype: str) -> float:
        m = self.means.get(stype, 0.0)
        s = self.stds.get(stype, 1.0)
        return y_norm * s + m


class PhysicsInformedDiodeHead(nn.Module if TORCH_AVAILABLE else object):
    """
    Physics-Informed Diode Head for JV Curves.
    Predicts diode parameters (Jsc, J0, n, Rs, Rsh) from material/parameter latent representation,
    evaluates explicit Shockley diode equation, and adds neural residual correction.
    """

    def __init__(self, in_features: int, fourier_dim: int = 16):
        if TORCH_AVAILABLE:
            super().__init__()
            self.param_net = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.GELU(),
                nn.Linear(64, 5)
            )
            self.residual_net = nn.Sequential(
                nn.Linear(in_features + 1 + fourier_dim, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
            self.vt = 0.02585  # Thermal voltage at 300K

    def forward(self, latent: torch.Tensor, v: torch.Tensor, fourier_v: torch.Tensor) -> torch.Tensor:
        params = self.param_net(latent)
        j_sc = F.softplus(params[:, 0:1]) + 0.1
        j0_log = params[:, 1:2]
        j0 = torch.exp(j0_log - 10.0) + 1e-12
        n = 1.0 + F.softplus(params[:, 2:3])
        r_s = F.softplus(params[:, 3:4]) * 0.01
        r_sh = 100.0 + F.softplus(params[:, 4:5]) * 100.0

        v_clamped = torch.clamp(v, -2.0, 3.0)
        exp_term = torch.exp(torch.clamp(v_clamped / (n * self.vt), -10.0, 15.0)) - 1.0
        j_diode = j_sc - j0 * exp_term - (v_clamped / (r_sh + 1e-5))

        res_input = torch.cat([latent, v, fourier_v], dim=-1)
        delta_j = self.residual_net(res_input)

        return j_diode + delta_j


class SolarV2MultiHeadSurrogate(nn.Module if TORCH_AVAILABLE else object):
    """
    V2 Multi-Head Solar Cell Curve Surrogate Neural Network.

    Architecture:
    1. Shared Material/Parameter Backbone MLP
    2. High-Frequency Fourier Coordinate Encoder for native x
    3. Dedicated Task Heads:
       - JV Head: PhysicsInformedDiodeHead
       - QE Head: High-Capacity MLP
       - RESP Head: High-Capacity MLP
    """

    def __init__(
        self,
        param_dim: int = 14,
        num_fourier_freqs: int = 8,
        hidden_dims: List[int] = [128, 128],
        dropout_rate: float = 0.05,
    ):
        if TORCH_AVAILABLE:
            super().__init__()
            self.fourier_encoder = FourierCoordinateEncoder(num_frequencies=num_fourier_freqs)
            fourier_dim = 2 * num_fourier_freqs

            layers = []
            in_dim = param_dim
            for hdim in hidden_dims:
                layers.append(nn.Linear(in_dim, hdim))
                layers.append(nn.LayerNorm(hdim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout_rate))
                in_dim = hdim
            self.backbone = nn.Sequential(*layers)
            latent_dim = hidden_dims[-1]

            self.jv_head = PhysicsInformedDiodeHead(in_features=latent_dim, fourier_dim=fourier_dim)

            self.qe_head = nn.Sequential(
                nn.Linear(latent_dim + 1 + fourier_dim, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

            self.resp_head = nn.Sequential(
                nn.Linear(latent_dim + 1 + fourier_dim, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

    def forward(self, features: torch.Tensor, stype: str = "JV") -> torch.Tensor:
        params = features[:, :14]
        x_raw = features[:, 14:15]
        fourier_x = self.fourier_encoder(x_raw)

        latent = self.backbone(params)

        if stype == "JV":
            return self.jv_head(latent, x_raw, fourier_x)
        elif stype == "QE":
            comb_input = torch.cat([latent, x_raw, fourier_x], dim=-1)
            return self.qe_head(comb_input)
        elif stype == "RESPONSIVITY":
            comb_input = torch.cat([latent, x_raw, fourier_x], dim=-1)
            return self.resp_head(comb_input)
        else:
            comb_input = torch.cat([latent, x_raw, fourier_x], dim=-1)
            return self.qe_head(comb_input)
