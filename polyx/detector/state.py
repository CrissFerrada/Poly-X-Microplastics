"""Estado compartido del Detector entre todas las páginas.

Hereda de QObject para emitir señales cuando cambia (las páginas se suscriben).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from PySide6.QtCore import QObject, Signal

from ..core.yolo_wrap import Detection, YoloModel


# ────────────────────────────────────────────────────────────────────
@dataclass
class ModelSlot:
    """Un slot de modelo en la pestaña 'Modelos' (hasta 3)."""
    alias: str = ""
    path: Optional[Path] = None      # ruta al .pt
    loaded: Optional[YoloModel] = None


@dataclass
class InferenceParams:
    conf: float = 0.25
    iou_nms: float = 0.45
    iou_tp: float = 0.50            # para análisis de errores
    # 1280 y no 640: los microplásticos ocupan ~12 px en fotos de 4096, y a 640
    # colapsan bajo el stride de la red, así que de fábrica no se detectaba nada.
    # Coincide con el perfil «Rápido» de la página de Parámetros.
    imgsz: int = 1280
    device: str = "0"               # "0", "cpu", ...
    um_per_px: float = 0.0          # 0 = no hay calibración
    size_min_um: float = 0.0        # 0 = sin filtro inferior
    size_max_um: float = 0.0        # 0 = sin filtro superior
    # ── Troceado automático ──
    # "auto" trocea sola la foto cuyo lado mayor pase de troceo_umbral_px, infiere
    # cada tile a resolución nativa y fusiona con NMS global. El umbral va a 2000
    # para que los recortes del estudio (1630 px de lado) entren de una pieza y
    # las placas completas (~3260) se troceen.
    troceo: str = "auto"            # "auto" | "siempre" | "nunca"
    troceo_umbral_px: int = 2000
    troceo_tile: int = 0            # 0 = derivar de min(umbral, imgsz)
    troceo_overlap: float = 0.25    # solape entre tiles; el NMS quita el duplicado


@dataclass
class ImageResult:
    """Resultado de inferir un modelo sobre una imagen."""
    image_path: Path
    model_idx: int
    predictions: List[Detection] = field(default_factory=list)
    gt: List[Detection] = field(default_factory=list)
    has_gt: bool = False
    tp: int = 0
    fp: int = 0
    fn: int = 0
    miscls: int = 0
    # bytes PNG de la imagen anotada combinada (predicción + GT) — preview/errores
    annotated_png: Optional[bytes] = None
    # bytes PNG solo con las predicciones del modelo (cajas de YOLO)
    pred_png: Optional[bytes] = None
    # bytes PNG solo con el Ground Truth (None si la imagen no tiene GT)
    gt_png: Optional[bytes] = None
    # veredicto del usuario tras revisión visual: None / "buena" / "mala"
    verdict: Optional[str] = None
    # geometría del troceado si la foto se partió; "" = se infirió de una pieza
    troceo: str = ""


# ────────────────────────────────────────────────────────────────────
class DetectorState(QObject):
    """Estado global del Detector. Las páginas leen/escriben aquí."""

    # Señales
    models_changed = Signal()
    images_changed = Signal()
    params_changed = Signal()
    run_progress = Signal(int, int, str)    # done, total, last_image
    run_started = Signal()
    run_finished = Signal()
    run_image_done = Signal(int, object)    # model_idx, ImageResult
    run_aborted = Signal()

    def __init__(self):
        super().__init__()
        self.model_slots: List[ModelSlot] = [ModelSlot(alias=f"Modelo {i+1}") for i in range(3)]
        self.images: List[Path] = []
        self.gt_folder: Optional[Path] = None
        self.params = InferenceParams()
        # results[model_idx] -> List[ImageResult]
        self.results: Dict[int, List[ImageResult]] = {}
        # run timestamp/folder
        self.run_dir: Optional[Path] = None
        self._running = False
        self._abort = False

    # ── helpers ──
    def active_models(self) -> List[ModelSlot]:
        return [s for s in self.model_slots if s.path is not None]

    def has_gt(self) -> bool:
        """¿Alguna imagen tiene GT?"""
        for r_list in self.results.values():
            if any(r.has_gt for r in r_list):
                return True
        return False

    def is_running(self) -> bool:
        return self._running

    def set_running(self, v: bool):
        self._running = v

    def request_abort(self):
        self._abort = True

    def consume_abort(self) -> bool:
        a = self._abort
        self._abort = False
        return a

    def reset_results(self):
        self.results = {}
        self.run_dir = None
