"""Detección de hardware y recomendaciones de imgsz / batch.

Heurística para responder rápidamente "¿cuál es el imgsz más alto al que
puedo entrenar sin OOM?". El usuario quiere por defecto entrenar al máximo
posible. Aquí calculamos un valor sensato a partir de la VRAM disponible.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    available: bool
    name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0


def detect_gpu() -> GPUInfo:
    try:
        import torch
        if not torch.cuda.is_available():
            return GPUInfo(available=False)
        idx = 0
        name = torch.cuda.get_device_name(idx)
        free, total = torch.cuda.mem_get_info(idx)
        return GPUInfo(
            available=True,
            name=name,
            vram_total_gb=total / (1024**3),
            vram_free_gb=free / (1024**3),
        )
    except Exception:
        return GPUInfo(available=False)


# Footprint aproximado de VRAM por arquitectura.
#   static_gb      = pesos + optimizador + gradientes (NO escala con batch)
#   act_gb_640_b1  = activaciones por imagen a imgsz=640 con AMP
#
# Calibrado contra mediciones reales (yolov8m 640/16 AMP ≈ 6 GB,
# 1280/8 AMP ≈ 10 GB) en RTX 30xx/40xx. Activaciones escalan con
# batch * (imgsz/640)^2.
_FOOTPRINT = {
    # size: (static_gb, act_gb_640_b1)
    "n": (0.30, 0.10),
    "s": (0.50, 0.18),
    "m": (1.00, 0.30),
    "l": (1.70, 0.50),
    "x": (2.80, 0.80),
}


def estimate_vram_gb(size: str, imgsz: int, batch: int, amp: bool = True) -> float:
    """Aproximación a la VRAM (GB) necesaria para entrenar.

    Fórmula: static_gb + act_gb_640_b1 * batch * (imgsz/640)^2 + overhead.
    El término estático (pesos+optim+grad) NO se multiplica por batch — ese
    era el bug que daba estimaciones absurdas (cientos de GB).
    """
    static_gb, act_gb = _FOOTPRINT.get(size, _FOOTPRINT["m"])
    if not amp:
        # FP32 ~ duplica activaciones y aumenta optimizador
        act_gb *= 1.8
        static_gb *= 1.5
    activations = act_gb * max(1, batch) * (imgsz / 640.0) ** 2
    overhead = 0.5   # CUDA/cuDNN/allocator
    return static_gb + activations + overhead


def recommend_max_imgsz(size: str, batch: int, vram_free_gb: float,
                        amp: bool = True,
                        safety: float = 0.85) -> int:
    """Devuelve el imgsz más alto en pasos de 32 que cabe en VRAM.

    Args:
        size: arquitectura ("n","s","m","l","x")
        batch: tamaño de batch
        vram_free_gb: VRAM disponible (GB)
        safety: factor de seguridad (0.85 = usar 85% del VRAM)
    """
    cap = vram_free_gb * safety
    candidates = [3840, 2560, 1920, 1600, 1280, 1024, 960, 800, 640, 512, 416]
    for sz in candidates:
        if estimate_vram_gb(size, sz, batch, amp) <= cap:
            return sz
    return 320


def recommend_batch(size: str, imgsz: int, vram_free_gb: float,
                    amp: bool = True, safety: float = 0.85) -> int:
    """Sugiere batch máximo para un imgsz dado."""
    cap = vram_free_gb * safety
    for b in (64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1):
        if estimate_vram_gb(size, imgsz, b, amp) <= cap:
            return b
    return 1


def humanize_gb(x: float) -> str:
    if x >= 10:
        return f"{x:.0f} GB"
    return f"{x:.1f} GB"
