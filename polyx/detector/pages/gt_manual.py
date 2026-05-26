"""Página 3 — GT manual.

Permite anotar manualmente bounding boxes sobre una imagen sin GT, para que
participe del análisis de errores. Es complementario al módulo Etiquetador
(que está pensado para datasets grandes); aquí es rápido y en-sitio.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QBrush, QImage, QFont, QPainterPath,
)
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox, QListWidget,
    QListWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsSimpleTextItem, QMessageBox, QFileDialog,
)

from ._base import DetectorPage
from ...core import theme as T
from ...core.yolo_wrap import Detection, read_yolo_txt, find_gt_for_image

DEFAULT_CLASSES = ["PET", "PP", "LDPE"]


# ────────────────────────────────────────────────────────────────────
class _AnnotCanvas(QGraphicsView):
    """Canvas que permite dibujar cajas con drag y mostrar las existentes."""
    boxes_changed = Signal()

    def __init__(self):
        super().__init__()
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.setStyleSheet(f"QGraphicsView {{ border: 1px solid {T.RULE}; background: #222; }}")
        self._pixmap_item = None
        self._image_size = (0, 0)
        self._boxes: List[Detection] = []   # en coords absolutas
        self._drawing = False
        self._start: Optional[QPointF] = None
        self._temp_rect: Optional[QGraphicsRectItem] = None
        self._active_class_id = 0
        self._class_names = list(DEFAULT_CLASSES)
        self.setDragMode(QGraphicsView.NoDrag)

    def set_class_names(self, names: List[str]):
        self._class_names = names

    def set_active_class(self, idx: int):
        self._active_class_id = idx

    def load_image(self, path: Path, existing: List[Detection]):
        self.scene_.clear()
        self._boxes = []
        pm = QPixmap(str(path))
        if pm.isNull():
            return
        self._image_size = (pm.width(), pm.height())
        self._pixmap_item = self.scene_.addPixmap(pm)
        self.setSceneRect(QRectF(0, 0, pm.width(), pm.height()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        for d in existing:
            self._add_visible_box(d)
        self._boxes = list(existing)
        self.boxes_changed.emit()

    def boxes(self) -> List[Detection]:
        return list(self._boxes)

    def clear_boxes(self):
        for it in list(self.scene_.items()):
            if isinstance(it, (QGraphicsRectItem, QGraphicsSimpleTextItem)):
                self.scene_.removeItem(it)
        self._boxes = []
        self.boxes_changed.emit()

    def _add_visible_box(self, d: Detection):
        color = self._class_color(d.class_id)
        rect = QGraphicsRectItem(QRectF(d.x1, d.y1, d.w, d.h))
        rect.setPen(QPen(color, 2))
        rect.setBrush(QBrush(QColor(0, 0, 0, 0)))
        self.scene_.addItem(rect)
        lbl = QGraphicsSimpleTextItem(self._class_names[d.class_id] if d.class_id < len(self._class_names) else str(d.class_id))
        lbl.setBrush(QBrush(color))
        f = QFont(T.APP_FONT_FAMILY, 9, QFont.Bold)
        lbl.setFont(f)
        lbl.setPos(d.x1, max(0, d.y1 - 16))
        self.scene_.addItem(lbl)

    def _class_color(self, cid: int) -> QColor:
        name = self._class_names[cid] if cid < len(self._class_names) else ""
        hex_ = T.CLASS_COLOR_HEX.get(name, "#33aaff")
        return QColor(hex_)

    # ── interacción de dibujo ──
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._pixmap_item is not None:
            self._drawing = True
            self._start = self.mapToScene(ev.position().toPoint())
            self._temp_rect = QGraphicsRectItem(QRectF(self._start, self._start))
            self._temp_rect.setPen(QPen(self._class_color(self._active_class_id), 2, Qt.DashLine))
            self.scene_.addItem(self._temp_rect)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drawing and self._temp_rect is not None:
            now = self.mapToScene(ev.position().toPoint())
            self._temp_rect.setRect(QRectF(self._start, now).normalized())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drawing and self._temp_rect is not None:
            r = self._temp_rect.rect()
            if r.width() >= 5 and r.height() >= 5:
                # convertir a Detection
                d = Detection(
                    class_id=self._active_class_id,
                    class_name=self._class_names[self._active_class_id] if self._active_class_id < len(self._class_names) else str(self._active_class_id),
                    conf=1.0,
                    x1=float(r.left()), y1=float(r.top()),
                    x2=float(r.right()), y2=float(r.bottom()),
                )
                self._boxes.append(d)
                self.scene_.removeItem(self._temp_rect)
                self._add_visible_box(d)
                self.boxes_changed.emit()
            else:
                self.scene_.removeItem(self._temp_rect)
            self._temp_rect = None
            self._drawing = False
            self._start = None
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            # eliminar última caja
            if self._boxes:
                self._boxes.pop()
                # redibujar
                pm_item = self._pixmap_item
                self.scene_.clear()
                if pm_item is not None:
                    self._pixmap_item = self.scene_.addPixmap(pm_item.pixmap())
                for d in self._boxes:
                    self._add_visible_box(d)
                self.boxes_changed.emit()
        super().keyPressEvent(ev)

    def resizeEvent(self, ev):
        if self._pixmap_item is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(ev)


# ────────────────────────────────────────────────────────────────────
class GTManualPage(DetectorPage):
    STEP_N = 3
    STEP_TITLE = "GT manual"
    STEP_DESCRIPTION = (
        "Para imágenes que aún no tienen ground truth, dibuja cajas a mano y guárdalas como "
        ".txt YOLO. Esto activará el análisis de errores en esas imágenes. "
        "Si tu dataset es grande, usa mejor el módulo Etiquetador."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Tarjeta: navegación + selector de clase ──
        c1, l1 = self.card("Anotación", "✏️")
        row = QHBoxLayout()
        row.setSpacing(8)

        self.lbl_image_name = QLabel("(sin imagen seleccionada)")
        self.lbl_image_name.setStyleSheet(f"color: {T.INK2}; font-weight: 600; border: none;")
        row.addWidget(self.lbl_image_name, 1)

        self.combo_class = QComboBox()
        self.combo_class.addItems(DEFAULT_CLASSES)
        self.combo_class.currentIndexChanged.connect(self._on_class_changed)
        row.addWidget(QLabel("Clase activa:"))
        row.addWidget(self.combo_class)
        l1.addLayout(row)

        # Lista de imágenes (compacta) + canvas
        body = QHBoxLayout()
        body.setSpacing(12)
        self.list = QListWidget()
        self.list.setMaximumWidth(280)
        self.list.itemSelectionChanged.connect(self._on_list_change)
        body.addWidget(self.list)

        self.canvas = _AnnotCanvas()
        self.canvas.set_class_names(DEFAULT_CLASSES)
        body.addWidget(self.canvas, 1)
        l1.addLayout(body)

        # Botonera inferior
        btns = QHBoxLayout()
        btns.setSpacing(8)
        b_save = QPushButton("💾  Guardar GT (.txt YOLO)")
        b_save.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        b_save.setCursor(Qt.PointingHandCursor)
        b_save.clicked.connect(self._save_current)
        btns.addWidget(b_save)

        b_clear = QPushButton("Limpiar cajas")
        b_clear.clicked.connect(self.canvas.clear_boxes)
        btns.addWidget(b_clear)
        btns.addStretch(1)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(f"color: {T.INK3}; border: none;")
        btns.addWidget(self.lbl_status)
        l1.addLayout(btns)

        self.body.addWidget(c1)

        # Suscribirse a cambios de imágenes
        self.state.images_changed.connect(self._refresh_list)
        self._refresh_list()

    # ──────────────────────────────────────────
    def _refresh_list(self):
        self.list.clear()
        for p in self.state.images:
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, str(p))
            gt = find_gt_for_image(p, self.state.gt_folder)
            if gt:
                it.setForeground(QColor(T.OK))
                it.setText(f"✓ {p.name}")
            self.list.addItem(it)

    def _on_class_changed(self, idx: int):
        self.canvas.set_active_class(idx)

    def _on_list_change(self):
        it = self.list.currentItem()
        if not it:
            return
        path = Path(it.data(Qt.UserRole))
        self.lbl_image_name.setText(path.name)
        # Cargar GT existente si hay
        existing: List[Detection] = []
        gt_txt = find_gt_for_image(path, self.state.gt_folder)
        if gt_txt:
            img = QImage(str(path))
            existing = read_yolo_txt(
                gt_txt, img.width(), img.height(),
                {i: n for i, n in enumerate(DEFAULT_CLASSES)},
            )
        self.canvas.load_image(path, existing)

    def _save_current(self):
        it = self.list.currentItem()
        if not it:
            QMessageBox.information(self, "GT manual", "Selecciona primero una imagen.")
            return
        path = Path(it.data(Qt.UserRole))
        boxes = self.canvas.boxes()
        if not boxes:
            QMessageBox.information(self, "GT manual", "No hay cajas para guardar.")
            return

        # Determinar carpeta destino
        if self.state.gt_folder:
            out_dir = self.state.gt_folder
        elif path.parent.name.lower() == "images":
            out_dir = path.parent.parent / "labels"
        else:
            out_dir = path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}.txt"

        img = QImage(str(path))
        W, H = img.width(), img.height()
        lines = []
        for d in boxes:
            cx = (d.x1 + d.x2) / 2 / W
            cy = (d.y1 + d.y2) / 2 / H
            w = (d.x2 - d.x1) / W
            h = (d.y2 - d.y1) / H
            lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        out_path.write_text("\n".join(lines), encoding="utf-8")

        # classes.txt junto al .txt
        cls_path = out_dir.parent / "classes.txt" if out_dir.name.lower() == "labels" else out_dir / "classes.txt"
        if not cls_path.exists():
            cls_path.write_text("\n".join(DEFAULT_CLASSES), encoding="utf-8")

        self.lbl_status.setText(f"Guardado: {out_path}")
        self._refresh_list()
