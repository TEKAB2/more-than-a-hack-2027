"""Light 3-frame TrackNet / UNet: heatmap + presence head (pure PyTorch).

Original TrackNet idea (Huang et al.): stack consecutive frames so a tiny fast
object leaves a temporal trail. Architecture here is a compact UNet written from
scratch (MIT-licensed repo code) — no Ultralytics / AGPL dependency.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TrackNetUNet(nn.Module):
    """9-channel input (3×RGB) → 1-channel heatmap + scalar presence logit."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.enc1 = _conv_block(9, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _conv_block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.presence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        presence = self.presence(b)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        heatmap = torch.sigmoid(self.head(d1))
        return heatmap, presence

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def heatmap_presence_loss(
    pred_heat: torch.Tensor,
    pred_pres: torch.Tensor,
    tgt_heat: torch.Tensor,
    tgt_pres: torch.Tensor,
    heat_weight: float = 1.0,
    pres_weight: float = 0.5,
) -> torch.Tensor:
    """BCE on heatmap + BCE-with-logits on presence."""
    # tgt_heat: (H, W) | (B, H, W) | (B, 1, H, W)
    if tgt_heat.dim() == 2:
        tgt_heat = tgt_heat.unsqueeze(0).unsqueeze(0)
    elif tgt_heat.dim() == 3:
        tgt_heat = tgt_heat.unsqueeze(1)
    if tgt_pres.dim() == 1:
        tgt_pres = tgt_pres.unsqueeze(1)
    elif tgt_pres.dim() == 0:
        tgt_pres = tgt_pres.view(1, 1)
    heat_loss = F.binary_cross_entropy(pred_heat, tgt_heat)
    pres_loss = F.binary_cross_entropy_with_logits(pred_pres, tgt_pres)
    return heat_weight * heat_loss + pres_weight * pres_loss


def save_checkpoint(
    path: str,
    model: TrackNetUNet,
    meta: Optional[Dict] = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "base": _infer_base(model),
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> Tuple[TrackNetUNet, Dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    base = int(payload.get("base", 32))
    model = TrackNetUNet(base=base)
    model.load_state_dict(payload["state_dict"])
    return model, dict(payload.get("meta") or {})


def _infer_base(model: TrackNetUNet) -> int:
    # First conv of enc1: Conv2d(9, base, ...)
    for module in model.enc1.modules():
        if isinstance(module, nn.Conv2d):
            return int(module.out_channels)
    return 32
