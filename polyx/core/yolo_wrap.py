"""Wrapper delgado sobre Ultralytics YOLO para inferencia + utilidades YOLO-txt."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import os

import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    # caja en coordenadas de píxeles absolutas (x1,y1,x2,y2)
    x1: float
    y1: float
    x2: float
    y2: float
    # tamaño equivalente (μm) si se calibró
    diam_um: Optional[float] = None
    area_um2: Optional[float] = None

    @property
    def w(self) -> float: return self.x2 - self.x1
    @property
    def h(self) -> float: return self.y2 - self.y1
    @property
    def cx(self) -> float: return 0.5 * (self.x1 + self.x2)
    @property
    def cy(self) -> float: return 0.5 * (self.y1 + self.y2)


class YoloModel:
    """Carga perezosa de Ultralytics YOLO. Permite múltiples modelos en paralelo."""

    def __init__(self, weights_path: str | Path, alias: str = ""):
        self.weights_path = str(weights_path)
        self.alias = alias or Path(self.weights_path).stem
        self._model = None
        self.names: Dict[int, str] = {}

    def load(self):
        if self._model is not None:
            return self
        from ultralytics import YOLO
        self._model = YOLO(self.weights_path)
        nm = getattr(self._model, "names", None) or {}
        self.names = dict(nm)
        return self

    def predict(self, image_path: str | Path, conf: float = 0.25, iou: float = 0.45,
                imgsz: int = 640, device: str = "0") -> List[Detection]:
        self.load()
        res = self._model.predict(
            source=str(image_path), conf=conf, iou=iou, imgsz=imgsz,
            device=device, verbose=False, save=False
        )
        out: List[Detection] = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
            out.append(Detection(
                class_id=int(cls),
                class_name=self.names.get(int(cls), str(int(cls))),
                conf=float(c),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            ))
        return out


def read_yolo_txt(txt_path: str | Path, img_w: int, img_h: int,
                  class_names: Dict[int, str]) -> List[Detection]:
    """Lee un .txt YOLO (class cx cy w h normalizados) y devuelve Detection con conf=1."""
    out: List[Detection] = []
    p = Path(txt_path)
    if not p.exists():
        return out
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            out.append(Detection(
                class_id=cls,
                class_name=class_names.get(cls, str(cls)),
                conf=1.0,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
    except Exception:
        pass
    return out


def find_gt_for_image(image_path: Path, gt_folder: Optional[Path]) -> Optional[Path]:
    """Busca el .txt GT junto a la imagen, en /labels hermana, o en gt_folder."""
    stem = image_path.stem
    candidates = []
    if gt_folder is not None:
        candidates.append(gt_folder / f"{stem}.txt")
    # mismo directorio
    candidates.append(image_path.parent / f"{stem}.txt")
    # hermana labels/
    if image_path.parent.name.lower() == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{stem}.txt")
    else:
        candidates.append(image_path.parent / "labels" / f"{stem}.txt")
    for c in candidates:
        if c.exists():
            return c
    return None


def compute_box_size_um(det: Detection, um_per_px: Optional[float]) -> None:
    """Asigna diam_um y area_um2 in-place si hay calibración."""
    if um_per_px is None or um_per_px <= 0:
        return
    area_px2 = det.w * det.h
    area_um2 = area_px2 * (um_per_px ** 2)
    # diámetro equivalente (círculo con misma área)
    diam_um = 2.0 * (area_um2 / np.pi) ** 0.5
    det.diam_um = diam_um
    det.area_um2 = area_um2
