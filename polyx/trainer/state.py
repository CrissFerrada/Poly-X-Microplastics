"""Estado compartido del Entrenador entre todas las páginas."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from PySide6.QtCore import QObject, Signal

from ..core.plataforma import dispositivo_disponible


# ────────────────────────────────────────────────────────────────────
# Presets de entrenamiento (Suave/Balanceado/Estricto)
# ────────────────────────────────────────────────────────────────────
PRESETS = {
    "Balanceado (recomendado)": {
        "epochs": 150, "imgsz": 1280, "batch": 8, "lr0": 0.01,
        "weight_decay": 0.0005, "momentum": 0.937, "warmup_epochs": 5,
        "close_mosaic": 15, "label_smoothing": 0.05, "patience": 50,
        "save_period": 20, "amp": True, "cos_lr": False, "cache": "disk",
        "aug_level": "Medio",
    },
    "Rápido (poco data)": {
        "epochs": 100, "imgsz": 960, "batch": 16, "lr0": 0.01,
        "weight_decay": 0.0005, "momentum": 0.937, "warmup_epochs": 3,
        "close_mosaic": 10, "label_smoothing": 0.0, "patience": 30,
        "save_period": 10, "amp": True, "cos_lr": False, "cache": "ram",
        "aug_level": "Fuerte",
    },
    "Largo (paper-quality)": {
        "epochs": 300, "imgsz": 1280, "batch": 8, "lr0": 0.005,
        "weight_decay": 0.0005, "momentum": 0.937, "warmup_epochs": 5,
        "close_mosaic": 20, "label_smoothing": 0.10, "patience": 100,
        "save_period": 25, "amp": True, "cos_lr": True, "cache": "disk",
        "aug_level": "Medio",
    },
    "Máximo imgsz (alta resolución)": {
        "epochs": 200, "imgsz": 1920, "batch": 4, "lr0": 0.01,
        "weight_decay": 0.0005, "momentum": 0.937, "warmup_epochs": 5,
        "close_mosaic": 15, "label_smoothing": 0.05, "patience": 60,
        "save_period": 20, "amp": True, "cos_lr": False, "cache": "disk",
        "aug_level": "Medio",
    },
    "Personalizado": {},  # marcador
}

AUG_LEVELS = {
    "Ninguno":  dict(hsv_h=0.0,  hsv_s=0.0,  hsv_v=0.0,  fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0),
    "Suave":    dict(hsv_h=0.01, hsv_s=0.5,  hsv_v=0.3,  fliplr=0.5, mosaic=0.5, mixup=0.0, copy_paste=0.0),
    "Medio":    dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.1),
    "Fuerte":   dict(hsv_h=0.02, hsv_s=0.9,  hsv_v=0.5,  fliplr=0.5, mosaic=1.0, mixup=0.2, copy_paste=0.3),
}


# ────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    family: str = "v8"      # "v8" o "v11"
    size: str = "m"          # n / s / m / l / x
    custom_weights: Optional[Path] = None
    preset_name: str = "Balanceado (recomendado)"
    # Entrena v8 y v11 seguidos con la MISMA configuracion, para poder atribuir
    # la diferencia de metricas a la arquitectura y no a los hiperparametros.
    comparar_familias: bool = False

    def familias_a_entrenar(self) -> list[str]:
        """Familias que se entrenaran en esta corrida, en orden."""
        return ["v8", "v11"] if self.comparar_familias else [self.family]

    def peso_de(self, familia: str) -> str:
        """Nombre del peso base de una familia concreta, mismo tamano."""
        fam = familia.lstrip("v") if familia != "v8" else familia
        return f"yolo{fam}{self.size}.pt"

    def base_weights_name(self) -> str:
        """Nombre del peso base tal como lo publica Ultralytics.

        Ojo con la nomenclatura, que es inconsistente y no perdona: la familia 8
        lleva "v" (yolov8m.pt) y la 11 no (yolo11m.pt). Construir el nombre
        pegando la familia tal cual daba "yolov11m.pt", que no existe en el
        catalogo y hacia fallar la descarga de toda la rama v11.
        """
        return self.peso_de(self.family)


@dataclass
class DatasetConfig:
    yaml_path: Optional[Path] = None
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    class_names: List[str] = field(default_factory=list)


@dataclass
class TrainParams:
    imgsz: int = 1280              # default ALTO (max practico) por requerimiento del usuario
    epochs: int = 150
    batch: int = 8
    lr0: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: int = 5
    close_mosaic: int = 15
    label_smoothing: float = 0.05
    patience: int = 50
    save_period: int = 20
    amp: bool = True
    cos_lr: bool = False
    cache: str = "disk"            # "disk", "ram", o "False"
    # Tras entrenar, reelegir el checkpoint evaluandolo solo sobre el sedimento
    # real de la validacion. best.pt sale del mAP global, que en un dataset mixto
    # lo dominan las placas de laboratorio: el dominio equivocado para detectar
    # en fotos de terreno. Ver polyx/trainer/dominio.py.
    elegir_por_dominio_real: bool = True
    # Ver el comentario equivalente en detector/state.py: "0" es CUDA y no
    # existe en un Mac. Se elige el mejor disponible al construir el estado.
    device: str = field(default_factory=dispositivo_disponible)
    workers: int = 8
    project_name: str = "polyx_train"
    run_name: str = ""             # vacío = auto-fecha


@dataclass
class AugParams:
    level: str = "Medio"
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1


# ────────────────────────────────────────────────────────────────────
@dataclass
class EpochMetrics:
    epoch: int
    box_loss: float = 0.0
    cls_loss: float = 0.0
    dfl_loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    lr: float = 0.0


# ────────────────────────────────────────────────────────────────────
class TrainerState(QObject):
    """Estado global del Entrenador."""

    # Señales
    model_changed = Signal()
    dataset_changed = Signal()
    params_changed = Signal()
    aug_changed = Signal()

    train_started = Signal()
    train_epoch = Signal(object)        # EpochMetrics
    train_log = Signal(str)             # línea de log de Ultralytics
    train_finished = Signal(str)        # ruta al best.pt
    train_aborted = Signal()
    train_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.model = ModelConfig()
        self.dataset = DatasetConfig()
        self.params = TrainParams()
        self.aug = AugParams()

        # historia de épocas durante el run actual
        self.history: List[EpochMetrics] = []
        self.best_map50: float = 0.0
        self.best_epoch: int = 0
        self.epochs_no_improve: int = 0

        self._running = False
        self._abort = False
        self.run_dir: Optional[Path] = None

    def is_running(self) -> bool: return self._running
    def set_running(self, v: bool): self._running = v
    def request_abort(self): self._abort = True
    def consume_abort(self) -> bool:
        a = self._abort; self._abort = False; return a

    def apply_preset(self, name: str):
        if name not in PRESETS or name == "Personalizado":
            return
        d = PRESETS[name]
        p = self.params
        a = self.aug
        for k, v in d.items():
            if k == "aug_level":
                a.level = v
                aug_vals = AUG_LEVELS.get(v, AUG_LEVELS["Medio"])
                for ak, av in aug_vals.items():
                    setattr(a, ak, av)
            elif hasattr(p, k):
                setattr(p, k, v)
        self.model.preset_name = name
        self.params_changed.emit()
        self.aug_changed.emit()
        self.model_changed.emit()
