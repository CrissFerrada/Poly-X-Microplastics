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
from PySide6.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QTableWidget, QTableWidgetItem,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsSimpleTextItem,
    QSplitter, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Colores Poly-X (General)
CLASS_COLOR = {
    "PET": QColor(255, 0, 0),      # rojo
    "PP": QColor(255, 140, 0),     # naranjo
    "LDPE": QColor(255, 215, 0),   # amarillo
    "PE": QColor(255, 215, 0),     # por si tu modelo usa PE
}

def bgr_to_qimage(img_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

@dataclass
class Detection:
    det_id: int            # ID global (por imagen)
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

class ClickableGraphicsView(QGraphicsView):
    clicked = Signal(QPointF)  # coordenadas escena

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            self.clicked.emit(pos)
        super().mousePressEvent(event)

class DetectionCanvas(QWidget):
    """
    Panel: imagen + cajas + callback click.
    Expone método para cargar detecciones y resaltar una.
    """
    box_clicked = Signal(int)  # det_id

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.view = ClickableGraphicsView()
        self.view.setScene(self.scene)
        self.view.setRenderHint(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.view)
        self.setLayout(layout)

        self.pix_item = None
        self.rect_items: Dict[int, QGraphicsRectItem] = {}
        self.text_items: Dict[int, QGraphicsSimpleTextItem] = {}

        self.detections: List[Detection] = []
        self.class_filter: Optional[set] = None  # set de class_name permitidos o None
        self.highlight_id: Optional[int] = None

        self.view.clicked.connect(self._on_click)

    def clear(self):
        self.scene.clear()
        self.pix_item = None
        self.rect_items.clear()
        self.text_items.clear()
        self.detections = []
        self.highlight_id = None

    def set_image(self, img_bgr: np.ndarray):
        self.clear()
        qimg = bgr_to_qimage(img_bgr)
        pix = QPixmap.fromImage(qimg)
        self.pix_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def set_filter(self, class_names: Optional[List[str]]):
        self.class_filter = set(class_names) if class_names else None

    def set_detections(self, dets: List[Detection], names_color: Dict[str, QColor], numbering_by_class: Dict[int, int]):
        """
        numbering_by_class: dict det_id->número mostrado (por clase/pestaña)
        """
        self.detections = dets
        # Dibujar cajas
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

            # etiqueta con número + conf
            n = numbering_by_class.get(d.det_id, d.det_id)
            label = f"{n}"
            text = QGraphicsSimpleTextItem(label)
            text.setBrush(QBrush(color))
            text.setZValue(3)
            text.setPos(d.x1, max(0, d.y1 - 16))
            self.scene.addItem(text)
            self.text_items[d.det_id] = text

    def highlight(self, det_id: Optional[int]):
        # Quita highlight previo
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
            # centrar
            self.view.centerOn(r)

    def _on_click(self, pos: QPointF):
        x, y = pos.x(), pos.y()
        # Buscar detección que contenga el punto (prioriza la de menor área para evitar cajas grandes tapando chicas)
        candidates: List[Tuple[float, int]] = []
        for d in self.detections:
            if self.class_filter is not None and d.class_name not in self.class_filter:
                continue
            if d.x1 <= x <= d.x2 and d.y1 <= y <= d.y2:
                candidates.append((d.area_px2, d.det_id))
        if not candidates:
            return
        candidates.sort(key=lambda t: t[0])
        chosen_id = candidates[0][1]
        self.box_clicked.emit(chosen_id)

class DetectionTable(QTableWidget):
    row_selected = Signal(int)  # det_id

    def __init__(self):
        super().__init__()
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels(["ID", "Clase", "Conf", "Xc", "Yc", "W", "H", "Área(px²)"])
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSortingEnabled(True)
        self._row_to_detid: Dict[int, int] = {}

        self.itemSelectionChanged.connect(self._emit_selected)

    def load(self, dets: List[Detection], class_filter: Optional[List[str]], numbering: Dict[int, int]):
        self.setSortingEnabled(False)
        self.clearContents()
        self.setRowCount(0)
        self._row_to_detid.clear()

        for d in dets:
            if class_filter is not None and d.class_name not in set(class_filter):
                continue
            row = self.rowCount()
            self.insertRow(row)
            shown_id = numbering.get(d.det_id, d.det_id)
            values = [
                str(shown_id),
                d.class_name,
                f"{d.conf:.3f}",
                f"{d.xc:.1f}",
                f"{d.yc:.1f}",
                f"{d.w:.1f}",
                f"{d.h:.1f}",
                f"{d.area_px2:.1f}",
            ]
            for col, v in enumerate(values):
                it = QTableWidgetItem(v)
                it.setData(Qt.UserRole, d.det_id)  # det_id real
                self.setItem(row, col, it)
            self._row_to_detid[row] = d.det_id

        self.resizeColumnsToContents()
        self.setSortingEnabled(True)

    def select_detid(self, det_id: int):
        # Busca fila que tenga ese det_id
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
    """
    Un tab: Canvas + Table
    """
    def __init__(self, tab_name: str, class_filter: Optional[List[str]]):
        super().__init__()
        self.tab_name = tab_name
        self.class_filter = class_filter

        self.canvas = DetectionCanvas()
        self.table = DetectionTable()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setSizes([800, 400])

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poly-X Detector® — Visor (1 imagen a la vez)")
        self.resize(1400, 800)

        self.model: Optional[YOLO] = None
        self.model_path: str = ""
        self.class_names: Dict[int, str] = {0:"PET", 1:"PP", 2:"LDPE"}
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

        # UI principal
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Barra superior
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

        btn_save = QPushButton("Guardar info (esta imagen)")
        btn_save.clicked.connect(self.save_current)

        btn_out = QPushButton("Elegir carpeta de salida")
        btn_out.clicked.connect(self.pick_output_dir)

        self.lbl_status = QLabel("Listo.")
        self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        top.addWidget(btn_model)
        top.addWidget(btn_open)
        top.addWidget(btn_open_folder)
        top.addWidget(btn_prev)
        top.addWidget(btn_next)
        top.addWidget(btn_run)
        top.addWidget(btn_save)
        top.addWidget(btn_out)
        top.addStretch(1)

        main_layout.addLayout(top)
        main_layout.addWidget(self.lbl_status)

        # Panel parámetros
        params = QGroupBox("Parámetros")
        form = QFormLayout(params)

        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0.01, 0.99)
        self.sp_conf.setSingleStep(0.01)
        self.sp_conf.setValue(0.40)

        self.sp_iou = QDoubleSpinBox()
        self.sp_iou.setRange(0.01, 0.99)
        self.sp_iou.setSingleStep(0.01)
        self.sp_iou.setValue(0.50)

        self.sp_imgsz = QSpinBox()
        self.sp_imgsz.setRange(320, 2048)
        self.sp_imgsz.setSingleStep(32)
        self.sp_imgsz.setValue(960)

        form.addRow("conf:", self.sp_conf)
        form.addRow("iou:", self.sp_iou)
        form.addRow("imgsz:", self.sp_imgsz)

        main_layout.addWidget(params)

        # Tabs
        self.tabs = QTabWidget()

        self.tab_general = TabPanel("General", class_filter=None)
        self.tab_pet = TabPanel("PET", class_filter=["PET"])
        self.tab_pp = TabPanel("PP", class_filter=["PP"])
        self.tab_ldpe = TabPanel("LDPE", class_filter=["LDPE", "PE"])  # tolera PE

        self.tabs.addTab(self._wrap_general_with_plot(), "General")
        self.tabs.addTab(self.tab_pet, "PET")
        self.tabs.addTab(self.tab_pp, "PP")
        self.tabs.addTab(self.tab_ldpe, "LDPE")

        main_layout.addWidget(self.tabs)

        # Linking bidireccional por tab
        self._wire_tab(self.tab_pet)
        self._wire_tab(self.tab_pp)
        self._wire_tab(self.tab_ldpe)

        # General: linking también
        self._wire_tab(self.tab_general)

    def _wrap_general_with_plot(self) -> QWidget:
        """
        General tab: arriba canvas+tabla, abajo gráfico barras.
        """
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)

        # splitter canvas/tabla
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tab_general.canvas)
        splitter.addWidget(self.tab_general.table)
        splitter.setSizes([800, 400])
        layout.addWidget(splitter)

        # plot barras
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(220)
        self.plot.setBackground(None)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setTitle("Conteo por polímero")
        layout.addWidget(self.plot)

        return wrapper

    def _wire_tab(self, tab: TabPanel):
        # Click imagen -> seleccionar tabla + highlight
        tab.canvas.box_clicked.connect(lambda det_id, t=tab: self._on_canvas_click(t, det_id))
        # Click tabla -> highlight + centrar
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

            # Leer names desde el modelo
            if hasattr(self.model, "names"):
                if isinstance(self.model.names, list):
                    self.class_names = {i: n for i, n in enumerate(self.model.names)}
                elif isinstance(self.model.names, dict):
                    self.class_names = self.model.names

            # Asegurar colores para nombres reales
            for n in self.class_names.values():
                if n not in self.class_colors:
                    self.class_colors[n] = QColor(0, 255, 0)

            self.set_status(f"Modelo cargado: {self.model_path} | Clases: {self.class_names}")
        except Exception as e:
            QMessageBox.critical(self, "Error cargando modelo", str(e))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
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
        if not self.current_folder_images:
            return
        if self.current_index <= 0:
            return
        self.current_index -= 1
        self._load_image(self.current_folder_images[self.current_index])

    def next_image(self):
        if not self.current_folder_images:
            return
        if self.current_index >= len(self.current_folder_images) - 1:
            return
        self.current_index += 1
        self._load_image(self.current_folder_images[self.current_index])

    def _load_image(self, path: str):
        self.current_image_path = os.path.abspath(path)
        img_bgr = cv2.imread(self.current_image_path)
        if img_bgr is None:
            QMessageBox.warning(self, "Error", "No pude leer la imagen.")
            return

        # Set image en todos los tabs
        self.tab_general.canvas.set_image(img_bgr)
        self.tab_pet.canvas.set_image(img_bgr)
        self.tab_pp.canvas.set_image(img_bgr)
        self.tab_ldpe.canvas.set_image(img_bgr)

        # Limpia detecciones/tabla/plot
        self.detections = []
        for tab in [self.tab_general, self.tab_pet, self.tab_pp, self.tab_ldpe]:
            tab.table.load([], tab.class_filter, {})
            tab.canvas.highlight(None)

        self._update_plot_counts({"PET": 0, "PP": 0, "LDPE": 0})

        self.set_status(f"Imagen cargada: {self.current_image_path}")

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
                    det_id=i+1,
                    class_id=cid,
                    class_name=cname,
                    conf=float(confs[i]),
                    x1=x1, y1=y1, x2=x2, y2=y2
                ))
        self.detections = dets

        # Numeración por clase (1..N dentro de cada pestaña)
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

        # Redibujar detecciones por tab
        # General: todas las clases
        self.tab_general.canvas.set_filter(None)
        self.tab_general.canvas.set_detections(dets, self.class_colors, self.numbering_general)
        self.tab_general.table.load(dets, None, self.numbering_general)

        # PET
        self.tab_pet.canvas.set_filter(["PET"])
        self.tab_pet.canvas.set_detections(dets, self.class_colors, self.numbering_pet)
        self.tab_pet.table.load(dets, ["PET"], self.numbering_pet)

        # PP
        self.tab_pp.canvas.set_filter(["PP"])
        self.tab_pp.canvas.set_detections(dets, self.class_colors, self.numbering_pp)
        self.tab_pp.table.load(dets, ["PP"], self.numbering_pp)

        # LDPE/PE
        self.tab_ldpe.canvas.set_filter(["LDPE", "PE"])
        self.tab_ldpe.canvas.set_detections(dets, self.class_colors, self.numbering_ldpe)
        self.tab_ldpe.table.load(dets, ["LDPE", "PE"], self.numbering_ldpe)

        # Plot conteos
        counts = {
            "PET": sum(1 for d in dets if d.class_name == "PET"),
            "PP": sum(1 for d in dets if d.class_name == "PP"),
            "LDPE": sum(1 for d in dets if d.class_name in ("LDPE", "PE")),
        }
        self._update_plot_counts(counts)

        self.set_status(f"Detectado: PET={counts['PET']} | PP={counts['PP']} | LDPE={counts['LDPE']} | Total={len(dets)}")

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

        # si no hay detecciones aún, igual permitimos guardar metadata (pero usualmente querrás detectar)
        ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
        ensure_dir(self.output_dir)

        base = os.path.splitext(os.path.basename(self.current_image_path))[0]
        out_json = os.path.join(self.output_dir, f"{base}_detections.json")
        out_csv = os.path.join(self.output_dir, f"{base}_detections.csv")

        # Guardar JSON
        payload = {
            "image_path": self.current_image_path,
            "model_path": self.model_path,
            "params": {"conf": float(self.sp_conf.value()), "iou": float(self.sp_iou.value()), "imgsz": int(self.sp_imgsz.value())},
            "detections": [d.__dict__ for d in self.detections],
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # Guardar CSV
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["det_id", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2", "xc", "yc", "w", "h", "area_px2"])
            for d in self.detections:
                w.writerow([d.det_id, d.class_id, d.class_name, d.conf, d.x1, d.y1, d.x2, d.y2, d.xc, d.yc, d.w, d.h, d.area_px2])

        # Guardar imágenes anotadas (general + por clase)
        img_bgr = cv2.imread(self.current_image_path)
        if img_bgr is not None:
            self._save_annotated_images(img_bgr, base)

        QMessageBox.information(self, "Guardado", f"Guardado:\n{out_json}\n{out_csv}\n(+ imágenes anotadas)")

    def _save_annotated_images(self, img_bgr: np.ndarray, base: str):
        # Dibujar con OpenCV (rápido) usando la numeración de cada vista
        def draw_for(filter_names: Optional[set], numbering: Dict[int, int], suffix: str):
            out = img_bgr.copy()
            for d in self.detections:
                if filter_names is not None and d.class_name not in filter_names:
                    continue
                color = self.class_colors.get(d.class_name, QColor(0,255,0))
                bgr = (color.blue(), color.green(), color.red())
                x1,y1,x2,y2 = map(int, (d.x1,d.y1,d.x2,d.y2))
                cv2.rectangle(out, (x1,y1), (x2,y2), bgr, 2)
                n = numbering.get(d.det_id, d.det_id)
                cv2.putText(out, str(n), (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
            cv2.imwrite(os.path.join(self.output_dir, f"{base}_{suffix}.png"), out)

        draw_for(None, self.numbering_general, "annotated_general")
        draw_for(set(["PET"]), self.numbering_pet, "annotated_PET")
        draw_for(set(["PP"]), self.numbering_pp, "annotated_PP")
        draw_for(set(["LDPE","PE"]), self.numbering_ldpe, "annotated_LDPE")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()