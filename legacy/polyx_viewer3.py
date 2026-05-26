import os
import sys
import json
import csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pyqtgraph as pg
from ultralytics import YOLO

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPixmap, QImage, QPen, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QTableWidget, QTableWidgetItem,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsSimpleTextItem,
    QGraphicsLineItem, QGraphicsEllipseItem, QSplitter, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Colores Poly-X
CLASS_COLOR = {
    "PET": QColor(255, 0, 0),      # rojo
    "PP": QColor(255, 140, 0),     # naranjo
    "LDPE": QColor(255, 215, 0),   # amarillo
    "PE": QColor(255, 215, 0),     # alias por si el modelo usa PE
}


def bgr_to_qimage(img_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def euclidean(a: QPointF, b: QPointF) -> float:
    dx = a.x() - b.x()
    dy = a.y() - b.y()
    return float((dx * dx + dy * dy) ** 0.5)


@dataclass
class Detection:
    det_id: int
    class_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def xc(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def yc(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area_px2(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def w_mm(self, px_per_mm: Optional[float]) -> Optional[float]:
        if not px_per_mm or px_per_mm <= 0:
            return None
        return self.w / px_per_mm

    def h_mm(self, px_per_mm: Optional[float]) -> Optional[float]:
        if not px_per_mm or px_per_mm <= 0:
            return None
        return self.h / px_per_mm

    def area_mm2_bbox(self, px_per_mm: Optional[float]) -> Optional[float]:
        if not px_per_mm or px_per_mm <= 0:
            return None
        return self.area_px2 / (px_per_mm ** 2)

    def deq_mm_bbox(self, px_per_mm: Optional[float]) -> Optional[float]:
        """Diámetro equivalente de círculo usando el área de la CAJA (aprox)."""
        a = self.area_mm2_bbox(px_per_mm)
        if a is None or a <= 0:
            return None
        return 2.0 * float((a / np.pi) ** 0.5)


class ClickableGraphicsView(QGraphicsView):
    clicked = Signal(QPointF)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            self.clicked.emit(pos)
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        zoom_in = 1.25
        zoom_out = 1 / zoom_in
        if event.angleDelta().y() > 0:
            factor = zoom_in
            self._zoom += 1
        else:
            factor = zoom_out
            self._zoom -= 1

        if self._zoom < -10:
            self._zoom = -10
            return

        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        self._zoom = 0
        if self.scene() is not None:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)


class DetectionCanvas(QWidget):
    box_clicked = Signal(int)        # det_id
    scene_clicked = Signal(QPointF)  # para calibración

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.view = ClickableGraphicsView()
        self.view.setScene(self.scene)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.setLayout(layout)

        self.pix_item = None

        # detecciones
        self.rect_items: Dict[int, QGraphicsRectItem] = {}
        self.text_items: Dict[int, QGraphicsSimpleTextItem] = {}
        self.detections: List[Detection] = []
        self.class_filter: Optional[set] = None
        self.highlight_id: Optional[int] = None

        # calibración overlay
        self.calib_line: Optional[QGraphicsLineItem] = None
        self.calib_text: Optional[QGraphicsSimpleTextItem] = None
        self.calib_mode: bool = False

        # círculo overlay
        self.calib_circle: Optional[QGraphicsEllipseItem] = None
        self.calib_center_item: Optional[QGraphicsEllipseItem] = None
        self.calib_pts_items: List[QGraphicsEllipseItem] = []

        self.view.clicked.connect(self._on_click)

    def set_image(self, img_bgr: np.ndarray):
        self.scene.clear()
        self.pix_item = None
        self.rect_items.clear()
        self.text_items.clear()
        self.detections = []
        self.highlight_id = None
        self.clear_calibration_full()

        qimg = bgr_to_qimage(img_bgr)
        pix = QPixmap.fromImage(qimg)
        self.pix_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def set_filter(self, class_names: Optional[List[str]]):
        self.class_filter = set(class_names) if class_names else None

    def clear_detections_overlay(self):
        """Borra solo cajas/textos (mantiene la imagen)."""
        for item in list(self.rect_items.values()):
            self.scene.removeItem(item)
        for item in list(self.text_items.values()):
            self.scene.removeItem(item)
        self.rect_items.clear()
        self.text_items.clear()
        self.detections = []
        self.highlight_id = None

    def set_detections(self, dets: List[Detection], names_color: Dict[str, QColor], numbering: Dict[int, int]):
        # CLAVE: limpiar overlays previos para evitar superposición
        self.clear_detections_overlay()

        self.detections = dets
        for d in dets:
            if self.class_filter is not None and d.class_name not in self.class_filter:
                continue

            color = names_color.get(d.class_name, QColor(0, 255, 0))
            pen = QPen(color, 2)
            brush = QBrush(QColor(color.red(), color.green(), color.blue(), 40))

            rect = QGraphicsRectItem(QRectF(d.x1, d.y1, d.w, d.h))
            rect.setPen(pen)
            rect.setBrush(brush)
            rect.setZValue(2)
            self.scene.addItem(rect)
            self.rect_items[d.det_id] = rect

            n = numbering.get(d.det_id, d.det_id)
            text = QGraphicsSimpleTextItem(str(n))
            text.setBrush(QBrush(color))
            text.setZValue(3)
            text.setPos(d.x1, max(0, d.y1 - 16))
            self.scene.addItem(text)
            self.text_items[d.det_id] = text

    def highlight(self, det_id: Optional[int]):
        if self.highlight_id is not None and self.highlight_id in self.rect_items:
            r = self.rect_items[self.highlight_id]
            pen = r.pen()
            pen.setWidth(2)
            r.setPen(pen)

        self.highlight_id = det_id
        if det_id is None:
            return
        if det_id in self.rect_items:
            r = self.rect_items[det_id]
            pen = r.pen()
            pen.setWidth(5)
            r.setPen(pen)
            self.view.centerOn(r)

    def set_calibration_mode(self, enabled: bool):
        self.calib_mode = enabled

    def _clear_line_text(self):
        if self.calib_line is not None:
            self.scene.removeItem(self.calib_line)
            self.calib_line = None
        if self.calib_text is not None:
            self.scene.removeItem(self.calib_text)
            self.calib_text = None

    def clear_calibration_full(self):
        # línea + texto
        self._clear_line_text()

        # círculo
        if self.calib_circle is not None:
            self.scene.removeItem(self.calib_circle)
            self.calib_circle = None
        if self.calib_center_item is not None:
            self.scene.removeItem(self.calib_center_item)
            self.calib_center_item = None

        # marcadores de clic
        for it in self.calib_pts_items:
            self.scene.removeItem(it)
        self.calib_pts_items = []

    def draw_calibration_overlay_line(self, p1: QPointF, p2: QPointF, label: str):
        self._clear_line_text()
        pen = QPen(QColor(0, 200, 255), 3)
        line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
        line.setPen(pen)
        line.setZValue(10)
        self.scene.addItem(line)
        self.calib_line = line

        txt = QGraphicsSimpleTextItem(label)
        txt.setBrush(QBrush(QColor(0, 200, 255)))
        txt.setZValue(11)
        txt.setPos(min(p1.x(), p2.x()), min(p1.y(), p2.y()) - 20)
        self.scene.addItem(txt)
        self.calib_text = txt

    def draw_click_marker(self, p: QPointF, color: QColor):
        pen = QPen(color, 3)
        r = 6
        it = self.scene.addEllipse(p.x() - r, p.y() - r, 2 * r, 2 * r, pen)
        it.setZValue(12)
        self.calib_pts_items.append(it)

    def draw_circle_overlay(self, cx: float, cy: float, r: float, label: str):
        # borra círculo previo pero mantiene marcadores
        if self.calib_circle is not None:
            self.scene.removeItem(self.calib_circle)
            self.calib_circle = None
        if self.calib_center_item is not None:
            self.scene.removeItem(self.calib_center_item)
            self.calib_center_item = None
        self._clear_line_text()

        pen = QPen(QColor(0, 200, 255), 3)
        circle = self.scene.addEllipse(cx - r, cy - r, 2 * r, 2 * r, pen)
        circle.setZValue(10)
        self.calib_circle = circle

        # centro
        pen2 = QPen(QColor(0, 200, 255), 3)
        rc = 5
        center = self.scene.addEllipse(cx - rc, cy - rc, 2 * rc, 2 * rc, pen2)
        center.setZValue(11)
        self.calib_center_item = center

        txt = QGraphicsSimpleTextItem(label)
        txt.setBrush(QBrush(QColor(0, 200, 255)))
        txt.setZValue(11)
        txt.setPos(cx - r, cy - r - 20)
        self.scene.addItem(txt)
        self.calib_text = txt

    def _on_click(self, pos: QPointF):
        self.scene_clicked.emit(pos)

        # calibración: en modo calibración no seleccionar cajas
        if self.calib_mode:
            return

        x, y = pos.x(), pos.y()
        candidates: List[Tuple[float, int]] = []
        for d in self.detections:
            if self.class_filter is not None and d.class_name not in self.class_filter:
                continue
            if d.x1 <= x <= d.x2 and d.y1 <= y <= d.y2:
                candidates.append((d.area_px2, d.det_id))
        if not candidates:
            return
        candidates.sort(key=lambda t: t[0])
        self.box_clicked.emit(candidates[0][1])


class DetectionTable(QTableWidget):
    row_selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSortingEnabled(True)
        self.itemSelectionChanged.connect(self._emit_selected)

    def load(self, dets: List[Detection], class_filter: Optional[List[str]],
             numbering: Dict[int, int], px_per_mm: Optional[float]):

        has_cal = bool(px_per_mm and px_per_mm > 0)
        headers = ["ID", "Clase", "Conf", "Xc(px)", "Yc(px)", "W(px)", "H(px)", "Área(px²)"]
        if has_cal:
            headers += ["W(mm)", "H(mm)", "Área_bbox(mm²)", "D_eq_bbox(mm)"]

        self.setSortingEnabled(False)
        self.clear()
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setRowCount(0)

        cfset = set(class_filter) if class_filter else None

        for d in dets:
            if cfset is not None and d.class_name not in cfset:
                continue
            row = self.rowCount()
            self.insertRow(row)
            shown_id = numbering.get(d.det_id, d.det_id)

            base_vals = [
                str(shown_id),
                d.class_name,
                f"{d.conf:.3f}",
                f"{d.xc:.1f}",
                f"{d.yc:.1f}",
                f"{d.w:.1f}",
                f"{d.h:.1f}",
                f"{d.area_px2:.1f}",
            ]
            extra_vals = []
            if has_cal:
                extra_vals = [
                    f"{d.w_mm(px_per_mm):.3f}" if d.w_mm(px_per_mm) is not None else "",
                    f"{d.h_mm(px_per_mm):.3f}" if d.h_mm(px_per_mm) is not None else "",
                    f"{d.area_mm2_bbox(px_per_mm):.4f}" if d.area_mm2_bbox(px_per_mm) is not None else "",
                    f"{d.deq_mm_bbox(px_per_mm):.4f}" if d.deq_mm_bbox(px_per_mm) is not None else "",
                ]

            values = base_vals + extra_vals
            for col, v in enumerate(values):
                it = QTableWidgetItem(v)
                it.setData(Qt.UserRole, d.det_id)
                self.setItem(row, col, it)

        self.resizeColumnsToContents()
        self.setSortingEnabled(True)

    def select_detid(self, det_id: int):
        for row in range(self.rowCount()):
            it = self.item(row, 0)
            if it and it.data(Qt.UserRole) == det_id:
                self.selectRow(row)
                self.scrollToItem(it)
                return

    def _emit_selected(self):
        items = self.selectedItems()
        if not items:
            return
        det_id = items[0].data(Qt.UserRole)
        if det_id is not None:
            self.row_selected.emit(int(det_id))


class TabPanel(QWidget):
    def __init__(self, class_filter: Optional[List[str]]):
        super().__init__()
        self.class_filter = class_filter
        self.canvas = DetectionCanvas()
        self.table = DetectionTable()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setSizes([950, 550])

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poly-X Detector® — Visor (1 imagen a la vez)")
        self.resize(1500, 850)

        self.model: Optional[YOLO] = None
        self.model_path: str = ""
        self.class_names: Dict[int, str] = {0: "PET", 1: "PP", 2: "LDPE"}
        self.class_colors: Dict[str, QColor] = CLASS_COLOR.copy()

        self.current_image_path: str = ""
        self.current_folder_images: List[str] = []
        self.current_index: int = -1

        self.detections: List[Detection] = []
        self.numbering_general: Dict[int, int] = {}
        self.numbering_pet: Dict[int, int] = {}
        self.numbering_pp: Dict[int, int] = {}
        self.numbering_ldpe: Dict[int, int] = {}

        self.output_dir: str = os.path.abspath("./polyx_outputs")

        # Calibración
        self.px_per_mm: Optional[float] = None
        self.calib_p1: Optional[Tuple[float, float]] = None
        self.calib_p2: Optional[Tuple[float, float]] = None
        self._calibrating: bool = False
        self._calib_on_tab: str = "General"

        # Modos calibración
        self._calib_mode = "none"   # "none" | "line" | "circle"
        self._circle_points: List[Tuple[float, float]] = []

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)

        # Top bar
        top = QHBoxLayout()
        btn_model = QPushButton("Cargar best.pt")
        btn_model.clicked.connect(self.load_model)

        btn_open = QPushButton("Abrir imagen")
        btn_open.clicked.connect(self.open_image)

        btn_open_folder = QPushButton("Abrir carpeta (navegar)")
        btn_open_folder.clicked.connect(self.open_folder)

        btn_prev = QPushButton("⟵ Anterior")
        btn_prev.clicked.connect(self.prev_image)

        btn_next = QPushButton("Siguiente ⟶")
        btn_next.clicked.connect(self.next_image)

        btn_run = QPushButton("Detectar")
        btn_run.clicked.connect(self.run_detection)

        btn_cal_line = QPushButton("Calibrar línea (2 clics)")
        btn_cal_line.clicked.connect(self.start_line_calibration)

        btn_cal_circle = QPushButton("Calibrar círculo (3 clics)")
        btn_cal_circle.clicked.connect(self.start_circle_calibration)

        btn_cal_reset = QPushButton("Rehacer calibración")
        btn_cal_reset.clicked.connect(self.reset_calibration)

        btn_save = QPushButton("Guardar info (esta imagen)")
        btn_save.clicked.connect(self.save_current)

        btn_out = QPushButton("Elegir carpeta de salida")
        btn_out.clicked.connect(self.pick_output_dir)

        top.addWidget(btn_model)
        top.addWidget(btn_open)
        top.addWidget(btn_open_folder)
        top.addWidget(btn_prev)
        top.addWidget(btn_next)
        top.addWidget(btn_run)
        top.addWidget(btn_cal_line)
        top.addWidget(btn_cal_circle)
        top.addWidget(btn_cal_reset)
        top.addWidget(btn_save)
        top.addWidget(btn_out)
        top.addStretch(1)

        self.lbl_status = QLabel("Listo.")
        self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        main.addLayout(top)
        main.addWidget(self.lbl_status)

        # Params
        params = QGroupBox("Parámetros (YOLO) + Calibración")
        form = QFormLayout(params)

        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0.01, 0.99)
        self.sp_conf.setSingleStep(0.01)
        self.sp_conf.setValue(0.40)
        self.sp_conf.setToolTip(
            "conf = umbral de confianza.\n"
            "Más alto => menos detecciones, más estrictas.\n"
            "Más bajo => más detecciones, más falsos positivos."
        )

        self.sp_iou = QDoubleSpinBox()
        self.sp_iou.setRange(0.01, 0.99)
        self.sp_iou.setSingleStep(0.01)
        self.sp_iou.setValue(0.50)
        self.sp_iou.setToolTip(
            "iou = umbral de NMS.\n"
            "Más bajo => elimina más cajas dobles.\n"
            "Más alto => permite más solapamiento."
        )

        self.sp_imgsz = QSpinBox()
        self.sp_imgsz.setRange(320, 2048)
        self.sp_imgsz.setSingleStep(32)
        self.sp_imgsz.setValue(960)
        self.sp_imgsz.setToolTip(
            "imgsz = reescalado para inferencia.\n"
            "Más alto => mejor para objetos pequeños, más lento."
        )

        self.sp_ref_mm = QDoubleSpinBox()
        self.sp_ref_mm.setRange(0.1, 100000.0)
        self.sp_ref_mm.setDecimals(2)
        self.sp_ref_mm.setSingleStep(1.0)
        self.sp_ref_mm.setValue(100.0)
        self.sp_ref_mm.setToolTip(
            "Línea: distancia real entre 2 puntos.\n"
            "Círculo: DIÁMETRO real de la placa (p.ej. 100 mm)."
        )

        form.addRow("conf:", self.sp_conf)
        form.addRow("iou:", self.sp_iou)
        form.addRow("imgsz:", self.sp_imgsz)
        form.addRow("Referencia (mm):", self.sp_ref_mm)

        self.lbl_cal = QLabel("(no definida)")
        form.addRow("Escala:", self.lbl_cal)

        self.lbl_help = QLabel(
            "Tips:\n"
            "• Línea: marca 2 puntos con distancia conocida (regla/patrón).\n"
            "• Círculo: marca 3 puntos del borde (mejor si se ve la placa).\n"
            "• Si cambias zoom/altura: calibra por imagen."
        )
        self.lbl_help.setStyleSheet("color: #444;")
        form.addRow(self.lbl_help)

        main.addWidget(params)

        # Tabs
        self.tabs = QTabWidget()
        self.tab_general = TabPanel(class_filter=None)
        self.tab_pet = TabPanel(class_filter=["PET"])
        self.tab_pp = TabPanel(class_filter=["PP"])
        self.tab_ldpe = TabPanel(class_filter=["LDPE", "PE"])

        self.tabs.addTab(self._wrap_general_with_plot(), "General")
        self.tabs.addTab(self.tab_pet, "PET")
        self.tabs.addTab(self.tab_pp, "PP")
        self.tabs.addTab(self.tab_ldpe, "LDPE")
        main.addWidget(self.tabs)

        # Linking
        self._wire_tab(self.tab_general)
        self._wire_tab(self.tab_pet)
        self._wire_tab(self.tab_pp)
        self._wire_tab(self.tab_ldpe)

        # Calibración: clicks por canvas
        self.tab_general.canvas.scene_clicked.connect(lambda p: self._on_scene_click("General", p))
        self.tab_pet.canvas.scene_clicked.connect(lambda p: self._on_scene_click("PET", p))
        self.tab_pp.canvas.scene_clicked.connect(lambda p: self._on_scene_click("PP", p))
        self.tab_ldpe.canvas.scene_clicked.connect(lambda p: self._on_scene_click("LDPE", p))

    def _wrap_general_with_plot(self) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tab_general.canvas)
        splitter.addWidget(self.tab_general.table)
        splitter.setSizes([1100, 600])
        left_layout.addWidget(splitter)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setTitle("Conteo por polímero")
        self.plot.setMinimumWidth(320)
        right_layout.addWidget(self.plot)

        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(left)
        main_split.addWidget(right)
        main_split.setSizes([1250, 320])

        layout.addWidget(main_split)
        return wrapper

    def _wire_tab(self, tab: TabPanel):
        tab.canvas.box_clicked.connect(lambda det_id, t=tab: self._on_canvas_click(t, det_id))
        tab.table.row_selected.connect(lambda det_id, t=tab: self._on_table_select(t, det_id))

    def _on_canvas_click(self, tab: TabPanel, det_id: int):
        tab.table.select_detid(det_id)
        tab.canvas.highlight(det_id)

    def _on_table_select(self, tab: TabPanel, det_id: int):
        tab.canvas.highlight(det_id)

    def set_status(self, msg: str):
        self.lbl_status.setText(msg)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecciona best.pt", "", "Weights (*.pt)")
        if not path:
            return
        try:
            self.model_path = os.path.abspath(path)
            self.model = YOLO(self.model_path)

            if hasattr(self.model, "names"):
                if isinstance(self.model.names, list):
                    self.class_names = {i: n for i, n in enumerate(self.model.names)}
                elif isinstance(self.model.names, dict):
                    self.class_names = self.model.names

            for n in self.class_names.values():
                if n not in self.class_colors:
                    self.class_colors[n] = QColor(0, 255, 0)

            self.set_status(f"Modelo cargado: {self.model_path} | Clases: {self.class_names}")
        except Exception as e:
            QMessageBox.critical(self, "Error cargando modelo", str(e))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        self.current_folder_images = []
        self.current_index = -1
        self._load_image(path)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Abrir carpeta de imágenes")
        if not folder:
            return
        imgs = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    imgs.append(os.path.join(root, f))
        imgs = sorted(imgs)
        if not imgs:
            QMessageBox.warning(self, "Sin imágenes", "No encontré imágenes en esa carpeta.")
            return
        self.current_folder_images = imgs
        self.current_index = 0
        self._load_image(imgs[0])

    def prev_image(self):
        if not self.current_folder_images or self.current_index <= 0:
            return
        self.current_index -= 1
        self._load_image(self.current_folder_images[self.current_index])

    def next_image(self):
        if not self.current_folder_images or self.current_index >= len(self.current_folder_images) - 1:
            return
        self.current_index += 1
        self._load_image(self.current_folder_images[self.current_index])

    def _load_image(self, path: str):
        self.current_image_path = os.path.abspath(path)
        img_bgr = cv2.imread(self.current_image_path)
        if img_bgr is None:
            QMessageBox.warning(self, "Error", "No pude leer la imagen.")
            return

        # reset calibración
        self.px_per_mm = None
        self.calib_p1 = None
        self.calib_p2 = None
        self._circle_points = []
        self._calibrating = False
        self._calib_mode = "none"
        self._set_cal_label()

        # apaga modo calibración + limpia overlays
        for tab in [self.tab_general, self.tab_pet, self.tab_pp, self.tab_ldpe]:
            tab.canvas.set_calibration_mode(False)
            tab.canvas.clear_calibration_full()

        # set image
        self.tab_general.canvas.set_image(img_bgr)
        self.tab_pet.canvas.set_image(img_bgr)
        self.tab_pp.canvas.set_image(img_bgr)
        self.tab_ldpe.canvas.set_image(img_bgr)

        # limpia detecciones
        self.detections = []
        for tab in [self.tab_general, self.tab_pet, self.tab_pp, self.tab_ldpe]:
            tab.table.load([], tab.class_filter, {}, self.px_per_mm)
            tab.canvas.highlight(None)

        self._update_plot_counts({"PET": 0, "PP": 0, "LDPE": 0})
        self.set_status(f"Imagen cargada: {self.current_image_path}")

    # ---------------- CALIBRACIÓN ----------------

    def reset_calibration(self):
        self.px_per_mm = None
        self.calib_p1 = None
        self.calib_p2 = None
        self._circle_points = []
        self._calibrating = False
        self._calib_mode = "none"
        self._set_cal_label()

        for tab in [self.tab_general, self.tab_pet, self.tab_pp, self.tab_ldpe]:
            tab.canvas.set_calibration_mode(False)
            tab.canvas.clear_calibration_full()

        self.set_status("Calibración reiniciada.")
        if self.detections:
            self._refresh_tables_only()

    def start_line_calibration(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Falta imagen", "Abre una imagen primero.")
            return

        ref_mm = float(self.sp_ref_mm.value())
        if ref_mm <= 0:
            QMessageBox.warning(self, "Referencia inválida", "La referencia en mm debe ser > 0.")
            return

        idx = self.tabs.currentIndex()
        tab_name = self.tabs.tabText(idx)

        self._calib_on_tab = tab_name
        self._calibrating = True
        self._calib_mode = "line"
        self.calib_p1 = None
        self.calib_p2 = None

        for name, tab in [("General", self.tab_general), ("PET", self.tab_pet), ("PP", self.tab_pp), ("LDPE", self.tab_ldpe)]:
            tab.canvas.set_calibration_mode(name == tab_name)
            if name == tab_name:
                tab.canvas.clear_calibration_full()

        self.set_status(f"Línea: haz 2 clics separados por {ref_mm:.2f} mm (pestaña actual).")

    def start_circle_calibration(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Falta imagen", "Abre una imagen primero.")
            return

        ref_mm = float(self.sp_ref_mm.value())
        if ref_mm <= 0:
            QMessageBox.warning(self, "Referencia inválida", "La referencia en mm debe ser > 0.")
            return

        idx = self.tabs.currentIndex()
        tab_name = self.tabs.tabText(idx)

        self._calib_on_tab = tab_name
        self._calibrating = True
        self._calib_mode = "circle"
        self._circle_points = []

        for name, tab in [("General", self.tab_general), ("PET", self.tab_pet), ("PP", self.tab_pp), ("LDPE", self.tab_ldpe)]:
            tab.canvas.set_calibration_mode(name == tab_name)
            if name == tab_name:
                tab.canvas.clear_calibration_full()

        self.set_status(f"Círculo: haz 3 clics sobre el BORDE (Ø real = {ref_mm:.2f} mm).")

    def _on_scene_click(self, tab_name: str, pos: QPointF):
        if not self._calibrating:
            return
        if tab_name != self._calib_on_tab:
            return

        ref_mm = float(self.sp_ref_mm.value())
        if ref_mm <= 0:
            return

        canvas = self._get_canvas_by_tab(tab_name)
        if canvas is None:
            return

        # ---- modo círculo ----
        if self._calib_mode == "circle":
            canvas.draw_click_marker(pos, QColor(255, 0, 255))
            self._circle_points.append((pos.x(), pos.y()))

            if len(self._circle_points) < 3:
                self.set_status(f"Círculo: punto {len(self._circle_points)} listo. Falta {3 - len(self._circle_points)}.")
                return

            (x1, y1), (x2, y2), (x3, y3) = self._circle_points[-3:]
            circle = self._circle_from_3_points(x1, y1, x2, y2, x3, y3)
            if circle is None:
                QMessageBox.warning(self, "Círculo inválido", "Los 3 puntos están casi en línea. Reintenta.")
                self._circle_points = []
                canvas.clear_calibration_full()
                self.set_status("Círculo reiniciado: vuelve a marcar 3 puntos en el borde.")
                return

            cx, cy, r_px = circle
            diameter_px = 2.0 * r_px
            # ref_mm aquí es DIÁMETRO real
            self.px_per_mm = diameter_px / ref_mm

            canvas.draw_circle_overlay(
                cx, cy, r_px,
                f"Ø {ref_mm:.2f} mm | Ø {diameter_px:.1f} px → {self.px_per_mm:.3f} px/mm"
            )

            self._finish_calibration(f"Calibración círculo lista: {self.px_per_mm:.3f} px/mm.")
            return

        # ---- modo línea ----
        if self._calib_mode == "line":
            if self.calib_p1 is None:
                self.calib_p1 = (pos.x(), pos.y())
                canvas.draw_click_marker(QPointF(self.calib_p1[0], self.calib_p1[1]), QColor(0, 200, 255))
                self.set_status("Línea: punto 1 listo. Ahora punto 2.")
                return

            if self.calib_p2 is None:
                self.calib_p2 = (pos.x(), pos.y())
                p1 = QPointF(self.calib_p1[0], self.calib_p1[1])
                p2 = QPointF(self.calib_p2[0], self.calib_p2[1])
                canvas.draw_click_marker(p2, QColor(255, 0, 255))

                dist_px = euclidean(p1, p2)
                if dist_px <= 1:
                    QMessageBox.warning(self, "Calibración inválida", "Los puntos están demasiado cerca.")
                    self.calib_p1 = None
                    self.calib_p2 = None
                    canvas.clear_calibration_full()
                    return

                self.px_per_mm = dist_px / ref_mm
                canvas.draw_calibration_overlay_line(
                    p1, p2, f"{ref_mm:.2f} mm | {dist_px:.1f} px → {self.px_per_mm:.3f} px/mm"
                )
                self._finish_calibration(f"Calibración línea lista: {self.px_per_mm:.3f} px/mm.")
                return

    def _finish_calibration(self, status_msg: str):
        self._calibrating = False
        self._calib_mode = "none"

        for tab in [self.tab_general, self.tab_pet, self.tab_pp, self.tab_ldpe]:
            tab.canvas.set_calibration_mode(False)

        self._set_cal_label()
        self.set_status(status_msg)

        if self.detections:
            self._refresh_tables_only()

    def _circle_from_3_points(self, x1, y1, x2, y2, x3, y3) -> Optional[Tuple[float, float, float]]:
        # Retorna (cx, cy, r) o None si casi colineales
        a = x1 - x2
        b = y1 - y2
        c = x1 - x3
        d = y1 - y3

        e = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0
        f = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0

        det = a * d - b * c
        if abs(det) < 1e-6:
            return None

        cx = (d * e - b * f) / det
        cy = (-c * e + a * f) / det

        r = float(((cx - x1) ** 2 + (cy - y1) ** 2) ** 0.5)
        if r <= 1:
            return None
        return (cx, cy, r)

    def _get_canvas_by_tab(self, tab_name: str) -> Optional[DetectionCanvas]:
        return {
            "General": self.tab_general.canvas,
            "PET": self.tab_pet.canvas,
            "PP": self.tab_pp.canvas,
            "LDPE": self.tab_ldpe.canvas,
        }.get(tab_name)

    def _set_cal_label(self):
        if self.px_per_mm and self.px_per_mm > 0:
            mm_per_px = 1.0 / self.px_per_mm
            self.lbl_cal.setText(f"{self.px_per_mm:.3f} px/mm  (≈ {mm_per_px:.6f} mm/px)")
        else:
            self.lbl_cal.setText("(no definida)")

    # ---------------- DETECCIÓN ----------------

    def run_detection(self):
        if self.model is None:
            QMessageBox.warning(self, "Falta modelo", "Carga primero un best.pt.")
            return
        if not self.current_image_path:
            QMessageBox.warning(self, "Falta imagen", "Abre una imagen primero.")
            return

        conf = float(self.sp_conf.value())
        iou = float(self.sp_iou.value())
        imgsz = int(self.sp_imgsz.value())

        self.set_status(f"Detectando... conf={conf} iou={iou} imgsz={imgsz}")

        try:
            res = self.model.predict(
                source=self.current_image_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )[0]
        except Exception as e:
            QMessageBox.critical(self, "Error en predict()", str(e))
            return

        dets: List[Detection] = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                cid = int(cls[i])
                cname = self.class_names.get(cid, str(cid))
                x1, y1, x2, y2 = map(float, boxes[i])
                dets.append(Detection(
                    det_id=i + 1,
                    class_id=cid,
                    class_name=cname,
                    conf=float(confs[i]),
                    x1=x1, y1=y1, x2=x2, y2=y2
                ))
        self.detections = dets

        # Numeración por pestaña
        self.numbering_general = {d.det_id: d.det_id for d in dets}
        self.numbering_pet = {}
        self.numbering_pp = {}
        self.numbering_ldpe = {}

        pet_n = pp_n = ldpe_n = 0
        for d in dets:
            if d.class_name == "PET":
                pet_n += 1
                self.numbering_pet[d.det_id] = pet_n
            if d.class_name == "PP":
                pp_n += 1
                self.numbering_pp[d.det_id] = pp_n
            if d.class_name in ("LDPE", "PE"):
                ldpe_n += 1
                self.numbering_ldpe[d.det_id] = ldpe_n

        # Render (limpia overlays anteriores dentro de set_detections)
        self.tab_general.canvas.set_filter(None)
        self.tab_general.canvas.set_detections(dets, self.class_colors, self.numbering_general)
        self.tab_general.table.load(dets, None, self.numbering_general, self.px_per_mm)

        self.tab_pet.canvas.set_filter(["PET"])
        self.tab_pet.canvas.set_detections(dets, self.class_colors, self.numbering_pet)
        self.tab_pet.table.load(dets, ["PET"], self.numbering_pet, self.px_per_mm)

        self.tab_pp.canvas.set_filter(["PP"])
        self.tab_pp.canvas.set_detections(dets, self.class_colors, self.numbering_pp)
        self.tab_pp.table.load(dets, ["PP"], self.numbering_pp, self.px_per_mm)

        self.tab_ldpe.canvas.set_filter(["LDPE", "PE"])
        self.tab_ldpe.canvas.set_detections(dets, self.class_colors, self.numbering_ldpe)
        self.tab_ldpe.table.load(dets, ["LDPE", "PE"], self.numbering_ldpe, self.px_per_mm)

        counts = {
            "PET": sum(1 for d in dets if d.class_name == "PET"),
            "PP": sum(1 for d in dets if d.class_name == "PP"),
            "LDPE": sum(1 for d in dets if d.class_name in ("LDPE", "PE")),
        }
        self._update_plot_counts(counts)

        self.set_status(
            f"Detectado: PET={counts['PET']} | PP={counts['PP']} | LDPE={counts['LDPE']} | Total={len(dets)}"
        )

    def _refresh_tables_only(self):
        self.tab_general.table.load(self.detections, None, self.numbering_general, self.px_per_mm)
        self.tab_pet.table.load(self.detections, ["PET"], self.numbering_pet, self.px_per_mm)
        self.tab_pp.table.load(self.detections, ["PP"], self.numbering_pp, self.px_per_mm)
        self.tab_ldpe.table.load(self.detections, ["LDPE", "PE"], self.numbering_ldpe, self.px_per_mm)

    def _update_plot_counts(self, counts: Dict[str, int]):
        self.plot.clear()
        labels = ["PET", "PP", "LDPE"]
        values = [counts.get("PET", 0), counts.get("PP", 0), counts.get("LDPE", 0)]
        x = np.arange(len(labels))
        bg = pg.BarGraphItem(x=x, height=values, width=0.6)
        self.plot.addItem(bg)
        ax = self.plot.getAxis("bottom")
        ax.setTicks([list(zip(x, labels))])
        self.plot.setLabel("left", "N partículas")

    # ---------------- GUARDADO ----------------

    def pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Selecciona carpeta de salida")
        if not d:
            return
        self.output_dir = os.path.abspath(d)
        self.set_status(f"Carpeta salida: {self.output_dir}")

    def save_current(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Falta imagen", "No hay imagen cargada.")
            return
        os.makedirs(self.output_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(self.current_image_path))[0]
        out_json = os.path.join(self.output_dir, f"{base}_detections.json")
        out_csv = os.path.join(self.output_dir, f"{base}_detections.csv")

        payload = {
            "image_path": self.current_image_path,
            "model_path": self.model_path,
            "params": {
                "conf": float(self.sp_conf.value()),
                "iou": float(self.sp_iou.value()),
                "imgsz": int(self.sp_imgsz.value()),
            },
            "calibration": {
                "reference_mm": float(self.sp_ref_mm.value()),
                "px_per_mm": self.px_per_mm,
                "mode": "circle_or_line_manual",
                "line_p1": self.calib_p1,
                "line_p2": self.calib_p2,
                "circle_points": self._circle_points,
            },
            "detections": [d.__dict__ for d in self.detections],
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        has_cal = bool(self.px_per_mm and self.px_per_mm > 0)
        headers = ["det_id", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2",
                   "xc", "yc", "w_px", "h_px", "area_px2"]
        if has_cal:
            headers += ["w_mm", "h_mm", "area_bbox_mm2", "deq_bbox_mm"]

        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for d in self.detections:
                row = [d.det_id, d.class_id, d.class_name, d.conf, d.x1, d.y1, d.x2, d.y2,
                       d.xc, d.yc, d.w, d.h, d.area_px2]
                if has_cal:
                    row += [
                        d.w_mm(self.px_per_mm),
                        d.h_mm(self.px_per_mm),
                        d.area_mm2_bbox(self.px_per_mm),
                        d.deq_mm_bbox(self.px_per_mm),
                    ]
                w.writerow(row)

        QMessageBox.information(self, "Guardado", f"Guardado:\n{out_json}\n{out_csv}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
