"""Estado del Etiquetador — imágenes, anotaciones, undo/redo."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal


CLASS_NAMES_DEFAULT = ["PET", "PP", "LDPE"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@dataclass
class BBox:
    class_id: int
    cx: float   # normalizado [0, 1]
    cy: float
    w: float
    h: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    @classmethod
    def from_yolo_line(cls, line: str) -> "BBox":
        parts = line.strip().split()
        return cls(int(parts[0]), float(parts[1]), float(parts[2]),
                   float(parts[3]), float(parts[4]))


class LabelerState(QObject):
    images_loaded      = Signal(list)   # List[Path]
    image_changed      = Signal(int)    # current_idx
    annotations_changed = Signal()
    classes_changed    = Signal(list)   # List[str]
    active_class_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self.images: list[Path] = []
        self.current_idx: int = -1
        self.class_names: list[str] = list(CLASS_NAMES_DEFAULT)
        self.active_class: int = 0

        self._annotations: dict[str, list[BBox]] = {}
        self._undo_stacks: dict[str, list[list[BBox]]] = {}
        self._redo_stacks: dict[str, list[list[BBox]]] = {}

    # ── Propiedades ───────────────────────────────────────────
    @property
    def current_image(self) -> Optional[Path]:
        if 0 <= self.current_idx < len(self.images):
            return self.images[self.current_idx]
        return None

    @property
    def current_boxes(self) -> list[BBox]:
        img = self.current_image
        return self._annotations.get(str(img), []) if img else []

    # ── Edición de anotaciones ───────────────────────────────
    def set_current_boxes(self, boxes: list[BBox], push_undo: bool = True):
        img = self.current_image
        if img is None:
            return
        key = str(img)
        if push_undo:
            self._push_undo(key)
        self._annotations[key] = list(boxes)
        self.annotations_changed.emit()

    def _push_undo(self, key: str):
        self._undo_stacks.setdefault(key, []).append(
            copy.deepcopy(self._annotations.get(key, []))
        )
        self._redo_stacks.setdefault(key, []).clear()

    def undo(self):
        img = self.current_image
        if not img:
            return
        key = str(img)
        stack = self._undo_stacks.get(key, [])
        if stack:
            self._redo_stacks.setdefault(key, []).append(
                copy.deepcopy(self._annotations.get(key, []))
            )
            self._annotations[key] = stack.pop()
            self.annotations_changed.emit()

    def redo(self):
        img = self.current_image
        if not img:
            return
        key = str(img)
        stack = self._redo_stacks.get(key, [])
        if stack:
            self._undo_stacks.setdefault(key, []).append(
                copy.deepcopy(self._annotations.get(key, []))
            )
            self._annotations[key] = stack.pop()
            self.annotations_changed.emit()

    # ── Navegación ───────────────────────────────────────────
    def load_images(self, folder: Path):
        imgs = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        self.images = imgs
        self._annotations.clear()
        self._undo_stacks.clear()
        self._redo_stacks.clear()
        self.images_loaded.emit(imgs)
        if imgs:
            self.current_idx = -1  # forzar goto a emitir image_changed
            self.goto(0)
        else:
            self.current_idx = -1

    def goto(self, idx: int):
        if not self.images:
            return
        idx = max(0, min(idx, len(self.images) - 1))
        if idx == self.current_idx:
            return
        self.save_current()
        self.current_idx = idx
        self._load_labels_for(self.images[idx])
        self.image_changed.emit(idx)
        self.annotations_changed.emit()

    def next_image(self):
        self.goto(self.current_idx + 1)

    def prev_image(self):
        self.goto(self.current_idx - 1)

    # ── Persistencia ─────────────────────────────────────────
    def save_current(self):
        img = self.current_image
        if img:
            self._save_labels_for(img, self.current_boxes)

    def _label_path_for(self, img: Path) -> Path:
        # Si imagen está en images/, guardar en labels/ hermana
        if img.parent.name.lower() == "images":
            lbl_dir = img.parent.parent / "labels"
        else:
            lbl_dir = img.parent
        lbl_dir.mkdir(parents=True, exist_ok=True)
        return lbl_dir / (img.stem + ".txt")

    def _save_labels_for(self, img: Path, boxes: list[BBox]):
        txt = self._label_path_for(img)
        content = "\n".join(b.to_yolo_line() for b in boxes)
        txt.write_text(content + "\n" if content else "", encoding="utf-8")

    def _load_labels_for(self, img: Path):
        key = str(img)
        if key in self._annotations:
            return
        txt = self._label_path_for(img)
        boxes: list[BBox] = []
        if txt.exists():
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        boxes.append(BBox.from_yolo_line(line))
                    except Exception:
                        pass
        self._annotations[key] = boxes

    def save_classes_txt(self):
        if not self.images:
            return
        root = self.images[0].parent
        if root.name.lower() == "images":
            root = root.parent
        (root / "classes.txt").write_text(
            "\n".join(self.class_names) + "\n", encoding="utf-8"
        )

    # ── Clases ───────────────────────────────────────────────
    def set_active_class(self, cls_id: int):
        self.active_class = cls_id
        self.active_class_changed.emit(cls_id)

    def set_class_names(self, names: list[str]):
        self.class_names = names
        self.classes_changed.emit(names)
