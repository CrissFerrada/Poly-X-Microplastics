"""Ventana principal del Visor — canvas (izq.) | panel de control (der.)."""
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton,
    QLabel, QFileDialog, QMessageBox, QInputDialog, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QDoubleSpinBox, QProgressDialog, QSizePolicy,
)

from ..core import theme as T
from ..core.widgets import LogoBadge, HLine
from .state import VisorState
from .canvas import VisorCanvas


class VisorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poly-X · Visor")
        self.resize(1420, 900)
        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG_SOFT}; }}")

        self.state = VisorState()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Canvas (centro/izquierda)
        canvas_wrap = QWidget()
        canvas_wrap.setStyleSheet(f"background: {T.BG_SOFT};")
        cw_lay = QVBoxLayout(canvas_wrap)
        cw_lay.setContentsMargins(0, 0, 0, 0)
        cw_lay.setSpacing(0)

        self.canvas = VisorCanvas(self.state)
        cw_lay.addWidget(self.canvas, 1)

        # Barra inferior de estado/calibración
        self.status_bar = QLabel("  Sin imagen  —  Sin calibración")
        self.status_bar.setFixedHeight(28)
        self.status_bar.setStyleSheet(
            f"background: {T.BG}; border-top: 1px solid {T.RULE}; "
            f"color: {T.INK2}; font-size: 9.5pt; padding: 0 12px;"
        )
        cw_lay.addWidget(self.status_bar)

        root.addWidget(canvas_wrap, 1)

        # Panel derecho
        right_scroll = QScrollArea()
        right_scroll.setFixedWidth(280)
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet(
            f"QScrollArea {{ border: none; border-left: 1px solid {T.RULE}; "
            f"background: {T.BG}; }}"
        )
        right_content = QWidget()
        right_content.setStyleSheet(f"background: {T.BG};")
        self._rv = QVBoxLayout(right_content)
        self._rv.setContentsMargins(14, 14, 14, 14)
        self._rv.setSpacing(10)
        right_scroll.setWidget(right_content)
        root.addWidget(right_scroll)

        self._build_right_panel()
        self._connect_signals()
        self._install_shortcuts()
        self.setAcceptDrops(True)

    # ── Panel derecho ─────────────────────────────────────────
    def _build_right_panel(self):
        rv = self._rv

        rv.addWidget(LogoBadge("POLY-X", "Visor"))
        rv.addSpacing(4)

        # ── Abrir imagen / carpeta ──
        lbl_open = QLabel("Imagen")
        lbl_open.setStyleSheet(f"font-weight: 600; font-size: 10pt;")
        rv.addWidget(lbl_open)

        row_open = QHBoxLayout()
        btn_img = QPushButton("📷  Imagen")
        btn_fol = QPushButton("📁  Carpeta")
        btn_img.clicked.connect(self._open_image)
        btn_fol.clicked.connect(self._open_folder)
        row_open.addWidget(btn_img)
        row_open.addWidget(btn_fol)
        rv.addLayout(row_open)

        self.lbl_img_name = QLabel("Sin imagen")
        self.lbl_img_name.setWordWrap(True)
        self.lbl_img_name.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        rv.addWidget(self.lbl_img_name)

        # Navegación (visible solo con carpeta)
        self.nav_frame = QFrame()
        self.nav_frame.setStyleSheet("QFrame { border: none; background: transparent; }")
        nav_lay = QHBoxLayout(self.nav_frame)
        nav_lay.setContentsMargins(0, 0, 0, 0)
        nav_lay.setSpacing(6)
        btn_prev = QPushButton("←")
        btn_next = QPushButton("→")
        btn_prev.setFixedWidth(36)
        btn_next.setFixedWidth(36)
        btn_prev.clicked.connect(self.state.prev_image)
        btn_next.clicked.connect(self.state.next_image)
        self.lbl_nav = QLabel("1 / 1")
        self.lbl_nav.setAlignment(Qt.AlignCenter)
        nav_lay.addWidget(btn_prev)
        nav_lay.addWidget(self.lbl_nav, 1)
        nav_lay.addWidget(btn_next)
        self.nav_frame.setVisible(False)
        rv.addWidget(self.nav_frame)

        rv.addWidget(HLine())

        # ── Modelo ──
        lbl_mod = QLabel("Modelo")
        lbl_mod.setStyleSheet(f"font-weight: 600; font-size: 10pt;")
        rv.addWidget(lbl_mod)

        self.lbl_model = QLabel("Sin modelo")
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setStyleSheet(
            f"color: {T.INK3}; font-size: 9pt; background: {T.BG_SOFT}; "
            f"border: 1px solid {T.RULE}; border-radius: 4px; padding: 4px;"
        )
        rv.addWidget(self.lbl_model)

        btn_load_model = QPushButton("📂  Cargar modelo…")
        btn_load_model.clicked.connect(self._load_model)
        rv.addWidget(btn_load_model)

        rv.addWidget(QLabel("Confianza mínima:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setDecimals(2)
        rv.addWidget(self.spin_conf)

        self.btn_detect = QPushButton("▶  Detectar")
        self.btn_detect.setObjectName("primary")
        self.btn_detect.clicked.connect(self._detect)
        rv.addWidget(self.btn_detect)

        rv.addWidget(HLine())

        # ── Calibración ──
        lbl_calib = QLabel("Calibración μm/píxel")
        lbl_calib.setStyleSheet(f"font-weight: 600; font-size: 10pt;")
        rv.addWidget(lbl_calib)

        row_calib = QHBoxLayout()
        self.btn_linea = QPushButton("📏  Línea")
        self.btn_circ  = QPushButton("⭕  Círculo")
        self.btn_linea.setCheckable(True)
        self.btn_circ.setCheckable(True)
        self.btn_linea.clicked.connect(lambda: self._start_calib("linea"))
        self.btn_circ.clicked.connect(lambda: self._start_calib("circulo"))
        row_calib.addWidget(self.btn_linea)
        row_calib.addWidget(self.btn_circ)
        rv.addLayout(row_calib)

        self.lbl_calib_hint = QLabel(
            "Haz clic en la imagen para marcar\npuntos de referencia."
        )
        self.lbl_calib_hint.setStyleSheet(f"color: {T.INK3}; font-size: 8.5pt;")
        self.lbl_calib_hint.setVisible(False)
        rv.addWidget(self.lbl_calib_hint)

        btn_cancel_calib = QPushButton("✕  Cancelar calibración")
        btn_cancel_calib.setStyleSheet(
            f"QPushButton {{ color: {T.WARN}; border-color: {T.WARN}; }}"
            f"QPushButton:hover {{ background: #fff8c5; }}"
        )
        btn_cancel_calib.clicked.connect(self.state.cancel_calib)
        self.btn_cancel_calib = btn_cancel_calib
        btn_cancel_calib.setVisible(False)
        rv.addWidget(btn_cancel_calib)

        self.lbl_calib_result = QLabel("📐  Sin calibrar")
        self.lbl_calib_result.setStyleSheet(f"color: {T.INK2}; font-size: 9.5pt;")
        rv.addWidget(self.lbl_calib_result)

        rv.addWidget(HLine())

        # ── Resultados ──
        lbl_res = QLabel("Resultados")
        lbl_res.setStyleSheet(f"font-weight: 600; font-size: 10pt;")
        rv.addWidget(lbl_res)

        rv.addWidget(QLabel("Filtrar por clase:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItem("Todas las clases")
        self.combo_filter.currentTextChanged.connect(self._on_filter_changed)
        rv.addWidget(self.combo_filter)

        self.lbl_det_count = QLabel("0 detecciones")
        self.lbl_det_count.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        rv.addWidget(self.lbl_det_count)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Clase", "Conf", "Ø(px)", "Ø(μm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setFixedHeight(220)
        self.table.setStyleSheet(f"font-size: 9pt;")
        rv.addWidget(self.table)

        rv.addWidget(HLine())

        # ── Guardar ──
        btn_save = QPushButton("💾  Guardar info actual")
        btn_save.setObjectName("primary")
        btn_save.clicked.connect(self._save)
        rv.addWidget(btn_save)

        self.lbl_status = QLabel("● Listo")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet(f"color: {T.OK}; font-size: 9pt;")
        rv.addWidget(self.lbl_status)

        rv.addStretch(1)

    # ── Conexiones y atajos ───────────────────────────────────
    def _connect_signals(self):
        self.state.image_loaded.connect(self._on_image_loaded)
        self.state.detections_changed.connect(self._on_detections_changed)
        self.state.calib_changed.connect(self._on_calib_changed)
        self.state.calib_pts_changed.connect(self._on_calib_pts_changed)
        self.state.calib_ready.connect(self._on_calib_ready)

    def _install_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+D"), self, self._detect)
        QShortcut(QKeySequence(Qt.Key_Left),  self, self.state.prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.state.next_image)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.state.cancel_calib)

    # ── Abrir archivos ────────────────────────────────────────
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )
        if path:
            self.state.load_single(Path(path))
            self.nav_frame.setVisible(False)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Abrir carpeta de imágenes")
        if folder:
            self.state.load_folder(Path(folder))
            total = len(self.state.images)
            if total > 1:
                self.nav_frame.setVisible(True)
                self._update_nav()

    def _on_image_loaded(self, path: Path):
        self.lbl_img_name.setText(path.name)
        self._update_nav()
        self._update_status_bar()
        self._set_status("● Imagen cargada", T.OK)

    def _update_nav(self):
        total = len(self.state.images)
        idx = self.state.current_idx
        self.lbl_nav.setText(f"{idx + 1} / {total}")

    # ── Modelo y detección ────────────────────────────────────
    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar modelo YOLO", "", "Modelo YOLO (*.pt)"
        )
        if not path:
            return
        try:
            from ..core.yolo_wrap import YoloModel
            self.state.model = YoloModel(path)
            self.lbl_model.setText(Path(path).name)
            self._set_status("● Modelo listo", T.OK)
        except Exception as e:
            QMessageBox.warning(self, "Error al cargar modelo", str(e))

    def _detect(self):
        if not self.state.model:
            QMessageBox.warning(self, "Sin modelo", "Carga un modelo .pt primero.")
            return
        img = self.state.current_image
        if img is None:
            QMessageBox.warning(self, "Sin imagen", "Abre una imagen primero.")
            return
        try:
            self._set_status("● Detectando…", T.ACCENT)
            dets = self.state.model.predict(
                str(img), conf=self.spin_conf.value(), device="cpu"
            )
            self.state.detections = dets
            self.state.detections_changed.emit(dets)
            self._set_status(f"● {len(dets)} detecciones", T.OK)
        except Exception as e:
            QMessageBox.warning(self, "Error en detección", str(e))
            self._set_status("● Error en detección", T.ERR)

    def _on_detections_changed(self, dets: list):
        self._rebuild_table(dets)
        self._update_filter_combo(dets)
        self.lbl_det_count.setText(f"{len(dets)} detección(es)")
        self._update_status_bar()

    def _rebuild_table(self, dets: list):
        self.table.setRowCount(0)
        for det in dets:
            row = self.table.rowCount()
            self.table.insertRow(row)
            w_px = det.x2 - det.x1
            h_px = det.y2 - det.y1
            diam_px = math.sqrt(w_px * h_px)
            um = self.state.px_to_um(diam_px)

            self.table.setItem(row, 0, QTableWidgetItem(det.class_name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{det.conf:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{diam_px:.1f}"))
            self.table.setItem(row, 3, QTableWidgetItem(
                f"{um:.1f}" if um is not None else "—"
            ))

    def _update_filter_combo(self, dets: list):
        self.combo_filter.blockSignals(True)
        prev = self.combo_filter.currentText()
        self.combo_filter.clear()
        self.combo_filter.addItem("Todas las clases")
        names = sorted({d.class_name for d in dets})
        self.combo_filter.addItems(names)
        idx = self.combo_filter.findText(prev)
        if idx >= 0:
            self.combo_filter.setCurrentIndex(idx)
        self.combo_filter.blockSignals(False)

    def _on_filter_changed(self, text: str):
        cls = None if text == "Todas las clases" else text
        self.canvas.set_filter_class(cls)

    # ── Calibración ───────────────────────────────────────────
    def _start_calib(self, mode: str):
        self.state.start_calib(mode)
        self.btn_linea.setChecked(mode == "linea")
        self.btn_circ.setChecked(mode == "circulo")
        self.lbl_calib_hint.setVisible(True)
        self.btn_cancel_calib.setVisible(True)
        if mode == "linea":
            self.lbl_calib_hint.setText(
                "Haz clic en 2 puntos sobre\nuna referencia conocida."
            )
        else:
            self.lbl_calib_hint.setText(
                "Haz clic en 3 puntos del borde\nde un objeto circular conocido."
            )
        self._set_status("● Modo calibración activo", T.WARN)

    def _on_calib_pts_changed(self, pts: list):
        if not pts:
            self.btn_linea.setChecked(False)
            self.btn_circ.setChecked(False)
            self.lbl_calib_hint.setVisible(False)
            self.btn_cancel_calib.setVisible(False)
        mode = self.state.calib_mode
        needed = 2 if mode == "linea" else 3
        self.lbl_calib_hint.setText(
            f"Haz clic en {needed - len(pts)} punto(s) más."
            if pts else self.lbl_calib_hint.text()
        )

    def _on_calib_ready(self):
        """Suficientes puntos recogidos → pedir tamaño real al usuario."""
        mode = self.state.calib_mode
        if mode == "linea":
            val, ok = QInputDialog.getDouble(
                self, "Calibración — Línea",
                "Ingresa el tamaño real de la línea marcada (μm):",
                value=10.0, min=0.001, max=1_000_000, decimals=3
            )
            if ok and val > 0:
                self.state.finish_calib_linea(val)
            else:
                self.state.cancel_calib()

        elif mode == "circulo":
            val, ok = QInputDialog.getDouble(
                self, "Calibración — Círculo",
                "Ingresa el diámetro real del objeto circular (μm):",
                value=10.0, min=0.001, max=1_000_000, decimals=3
            )
            if ok and val > 0:
                self.state.finish_calib_circulo(val)
            else:
                self.state.cancel_calib()

    def _on_calib_changed(self, um_per_px: float, mode_name: str):
        self.lbl_calib_result.setText(f"📐  {um_per_px:.4f} μm/px  ({mode_name})")
        self.lbl_calib_result.setStyleSheet(f"color: {T.OK}; font-size: 9.5pt; font-weight: 600;")
        self._set_status(f"● Calibración aplicada: {um_per_px:.4f} μm/px", T.OK)
        self._update_status_bar()
        # Refrescar tabla con nuevas unidades μm
        if self.state.detections:
            self._rebuild_table(self.state.detections)

    def _update_status_bar(self):
        img = self.state.current_image
        img_txt = img.name if img else "Sin imagen"
        if self.state.um_per_px > 0:
            calib_txt = f"📐 {self.state.um_per_px:.4f} μm/px"
        else:
            calib_txt = "Sin calibración"
        n_det = len(self.state.detections)
        det_txt = f"  —  {n_det} det." if n_det else ""
        self.status_bar.setText(f"  {img_txt}  —  {calib_txt}{det_txt}")

    # ── Guardar ───────────────────────────────────────────────
    def _save(self):
        img = self.state.current_image
        if img is None:
            QMessageBox.warning(self, "Sin imagen", "No hay imagen abierta.")
            return

        # Elegir carpeta destino
        out_dir = QFileDialog.getExistingDirectory(self, "Carpeta de salida")
        if not out_dir:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(out_dir) / f"visor_{img.stem}_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._save_annotated_image(img, save_dir)
            self._save_csv(save_dir)
            self._save_json(img, save_dir)
            self._set_status(f"● Guardado en {save_dir.name}", T.OK)
            QMessageBox.information(self, "Guardado",
                f"Archivos guardados en:\n{save_dir}")
        except Exception as e:
            QMessageBox.warning(self, "Error al guardar", str(e))
            self._set_status("● Error al guardar", T.ERR)

    def _save_annotated_image(self, img: Path, save_dir: Path):
        import cv2
        import numpy as np
        frame = cv2.imread(str(img))
        if frame is None:
            return

        CLASS_COLORS_BGR = [
            (47,  52, 227),   # PET  → azul (BGR)
            (0,  140, 255),   # PP   → naranjo
            (0,  215, 255),   # LDPE → amarillo
            (216, 180,   0),
            (228,  202,  72),
            (182, 119,   0),
            (138,  62,   2),
            (139,  45, 118),
        ]
        for det in self.state.detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            color = CLASS_COLORS_BGR[det.class_id % len(CLASS_COLORS_BGR)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            w_px = det.x2 - det.x1
            h_px = det.y2 - det.y1
            diam_px = math.sqrt(w_px * h_px)
            um = self.state.px_to_um(diam_px)
            label = (f"{det.class_name} {um:.1f}um"
                     if um is not None else f"{det.class_name} {diam_px:.0f}px")
            cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out_path = save_dir / f"{img.stem}_anotada.png"
        cv2.imwrite(str(out_path), frame)

    def _save_csv(self, save_dir: Path):
        import csv
        out = save_dir / "detecciones.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clase", "conf", "x1", "y1", "x2", "y2",
                         "diam_px", "diam_um", "area_um2"])
            for det in self.state.detections:
                wp = det.x2 - det.x1
                hp = det.y2 - det.y1
                diam_px = math.sqrt(wp * hp)
                um = self.state.px_to_um(diam_px)
                area_um2 = (self.state.um_per_px ** 2 * wp * hp
                            if self.state.um_per_px > 0 else "")
                w.writerow([
                    det.class_name, f"{det.conf:.4f}",
                    f"{det.x1:.1f}", f"{det.y1:.1f}",
                    f"{det.x2:.1f}", f"{det.y2:.1f}",
                    f"{diam_px:.2f}",
                    f"{um:.2f}" if um is not None else "",
                    f"{area_um2:.2f}" if area_um2 != "" else "",
                ])

    def _save_json(self, img: Path, save_dir: Path):
        um = self.state.um_per_px
        summary = {
            "imagen": img.name,
            "timestamp": datetime.now().isoformat(),
            "um_per_px": um if um > 0 else None,
            "n_detecciones": len(self.state.detections),
            "por_clase": {},
        }
        for det in self.state.detections:
            entry = summary["por_clase"].setdefault(det.class_name, {"n": 0})
            entry["n"] += 1
        out = save_dir / "resumen.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Drag & Drop ───────────────────────────────────────────
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            for u in ev.mimeData().urls():
                p = u.toLocalFile().lower()
                if p.endswith(".pt") or any(p.endswith(e) for e in
                        (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):
                    ev.acceptProposedAction()
                    return
        ev.ignore()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() == ".pt":
                try:
                    from ..core.yolo_wrap import YoloModel
                    self.state.model = YoloModel(str(path))
                    self.lbl_model.setText(path.name)
                    self._set_status("● Modelo cargado", T.OK)
                except Exception as e:
                    self._set_status(f"● Error: {e}", T.ERR)
            elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
                self.state.load_single(path)
                self.nav_frame.setVisible(False)
        ev.acceptProposedAction()

    def _set_status(self, text: str, color: str = T.OK):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 9pt;")
