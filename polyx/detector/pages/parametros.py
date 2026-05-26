"""Página 4 — Parámetros de inferencia + calibración óptica + filtros."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QLineEdit, QFrame,
)

from ._base import DetectorPage
from ...core import theme as T


def _hint(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
    l.setWordWrap(True)
    return l


class ParametrosPage(DetectorPage):
    STEP_N = 4
    STEP_TITLE = "Parámetros"
    STEP_DESCRIPTION = (
        "Configura confianza, IoU, calibración óptica y filtros. La calibración μm/px es "
        "opcional pero recomendada para reportes científicos."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Inferencia ──
        c1, l1 = self.card("Inferencia", "⚙️")
        g = QGridLayout()
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(10)

        # Confianza
        g.addWidget(QLabel("Confianza mín. (conf):"), 0, 0)
        self.sb_conf = QDoubleSpinBox()
        self.sb_conf.setRange(0.01, 1.0); self.sb_conf.setSingleStep(0.05)
        self.sb_conf.setDecimals(2); self.sb_conf.setValue(state.params.conf)
        self.sb_conf.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_conf, 0, 1)
        g.addWidget(_hint("0.05–0.95. Más alta = menos detecciones (con falsos positivos)."), 0, 2)

        # IoU NMS
        g.addWidget(QLabel("IoU NMS:"), 1, 0)
        self.sb_iou_nms = QDoubleSpinBox()
        self.sb_iou_nms.setRange(0.1, 0.9); self.sb_iou_nms.setSingleStep(0.05)
        self.sb_iou_nms.setDecimals(2); self.sb_iou_nms.setValue(state.params.iou_nms)
        self.sb_iou_nms.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_iou_nms, 1, 1)
        g.addWidget(_hint("0.30–0.70. Suprime cajas superpuestas."), 1, 2)

        # imgsz
        g.addWidget(QLabel("Tamaño imagen (imgsz):"), 2, 0)
        self.sb_imgsz = QSpinBox()
        self.sb_imgsz.setRange(320, 1920); self.sb_imgsz.setSingleStep(32)
        self.sb_imgsz.setValue(state.params.imgsz)
        self.sb_imgsz.valueChanged.connect(self._on_change)
        g.addWidget(self.sb_imgsz, 2, 1)
        g.addWidget(_hint("Debe coincidir con el del entrenamiento. Típico: 640 o 1280."), 2, 2)

        # device
        g.addWidget(QLabel("Device (0/cpu):"), 3, 0)
        self.ed_device = QLineEdit(state.params.device)
        self.ed_device.editingFinished.connect(self._on_change)
        g.addWidget(self.ed_device, 3, 1)
        g.addWidget(_hint("'0' = primera GPU. 'cpu' = CPU. '0,1' = GPU 0 y 1."), 3, 2)

        l1.addLayout(g)
        self.body.addWidget(c1)

        # ── Análisis de errores ──
        c2, l2 = self.card("Análisis de errores (si hay GT)", "🎯")
        g2 = QGridLayout()
        g2.setHorizontalSpacing(24); g2.setVerticalSpacing(10)
        g2.addWidget(QLabel("IoU para emparejar TP:"), 0, 0)
        self.sb_iou_tp = QDoubleSpinBox()
        self.sb_iou_tp.setRange(0.1, 0.95); self.sb_iou_tp.setSingleStep(0.05)
        self.sb_iou_tp.setDecimals(2); self.sb_iou_tp.setValue(state.params.iou_tp)
        self.sb_iou_tp.valueChanged.connect(self._on_change)
        g2.addWidget(self.sb_iou_tp, 0, 1)
        g2.addWidget(_hint("Estándar COCO: 0.50. Más estricto: 0.75."), 0, 2)
        l2.addLayout(g2)
        l2.addWidget(_hint(
            "TP: clase correcta + IoU ≥ umbral.  FP: predicción sin GT cercano.  "
            "FN: GT no detectado.  MISCLS: IoU alto pero clase distinta."
        ))
        self.body.addWidget(c2)

        # ── Calibración óptica ──
        c3, l3 = self.card("Calibración óptica (tamaño de partícula)", "📐")
        g3 = QGridLayout()
        g3.setHorizontalSpacing(24); g3.setVerticalSpacing(10)
        g3.addWidget(QLabel("μm por píxel:"), 0, 0)
        self.sb_umpx = QDoubleSpinBox()
        self.sb_umpx.setRange(0.0, 10.0); self.sb_umpx.setSingleStep(0.01)
        self.sb_umpx.setDecimals(4); self.sb_umpx.setValue(state.params.um_per_px)
        self.sb_umpx.valueChanged.connect(self._on_change)
        g3.addWidget(self.sb_umpx, 0, 1)
        g3.addWidget(_hint(
            "0 = sin medición. Ej: objetivo 40× y CMOS calibrado → ~0.244 μm/px."
        ), 0, 2)
        l3.addLayout(g3)
        l3.addWidget(_hint(
            "El detector calcula área y diámetro equivalente (de círculo con misma área) "
            "para cada partícula detectada. Si hay calibración, también en μm y μm²."
        ))
        self.body.addWidget(c3)

        # ── Filtro por tamaño ──
        c4, l4 = self.card("Filtro por tamaño (opcional)", "📏")
        g4 = QGridLayout()
        g4.setHorizontalSpacing(24); g4.setVerticalSpacing(10)
        g4.addWidget(QLabel("Tamaño mín (μm):"), 0, 0)
        self.sb_min = QDoubleSpinBox(); self.sb_min.setRange(0.0, 10000.0)
        self.sb_min.setDecimals(2); self.sb_min.setValue(state.params.size_min_um)
        self.sb_min.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_min, 0, 1)

        g4.addWidget(QLabel("Tamaño máx (μm):"), 0, 2)
        self.sb_max = QDoubleSpinBox(); self.sb_max.setRange(0.0, 10000.0)
        self.sb_max.setDecimals(2); self.sb_max.setValue(state.params.size_max_um)
        self.sb_max.valueChanged.connect(self._on_change)
        g4.addWidget(self.sb_max, 0, 3)
        g4.addWidget(_hint("0 = sin filtro. Aplica sobre el diámetro equivalente."), 1, 0, 1, 4)
        l4.addLayout(g4)
        self.body.addWidget(c4)

    def _on_change(self, *_):
        p = self.state.params
        p.conf = float(self.sb_conf.value())
        p.iou_nms = float(self.sb_iou_nms.value())
        p.iou_tp = float(self.sb_iou_tp.value())
        p.imgsz = int(self.sb_imgsz.value())
        p.device = self.ed_device.text().strip() or "0"
        p.um_per_px = float(self.sb_umpx.value())
        p.size_min_um = float(self.sb_min.value())
        p.size_max_um = float(self.sb_max.value())
        self.state.params_changed.emit()
