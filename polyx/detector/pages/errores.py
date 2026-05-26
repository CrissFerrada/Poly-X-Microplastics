"""Página 7 — Errores (FP / FN / MISCLS). Lista filtrable + preview."""
from __future__ import annotations
import os
from pathlib import Path

from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton,
)

from ._base import DetectorPage
from ...core import theme as T
from ...core.metrics import match_image


class ErroresPage(DetectorPage):
    STEP_N = 7
    STEP_TITLE = "Errores"
    STEP_DESCRIPTION = (
        "Lista de cajas problemáticas (solo si hay Ground Truth). "
        "FP: detección sin GT cercano. FN: GT no detectado. MISCLS: bien localizado, mala clase."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Filtros ──
        c1, l1 = self.card("Filtros", "🔎")
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(QLabel("Tipo:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Todos", "FP", "FN", "MISCLS"])
        self.combo_type.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.combo_type)

        row.addWidget(QLabel("Modelo:"))
        self.combo_model = QComboBox()
        self.combo_model.addItem("Todos", -1)
        self.combo_model.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.combo_model)

        row.addStretch(1)
        self.lbl_count = QLabel("0 errores")
        self.lbl_count.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        row.addWidget(self.lbl_count)
        l1.addLayout(row)
        self.body.addWidget(c1)

        # ── Tabla de errores ──
        c2, l2 = self.card("Cajas con error", "⚠")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Modelo", "Imagen", "Tipo", "Clase GT", "Clase Pred", "Conf"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(280)
        self.table.cellDoubleClicked.connect(self._open_image)
        l2.addWidget(self.table)
        self.body.addWidget(c2)

        # Conectar
        self.state.run_finished.connect(self.refresh)
        self.state.models_changed.connect(self._refresh_model_combo)

    def _refresh_model_combo(self):
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        self.combo_model.addItem("Todos", -1)
        for i, slot in enumerate(self.state.model_slots):
            if slot.path is not None:
                self.combo_model.addItem(slot.alias, i)
        self.combo_model.blockSignals(False)

    def refresh(self):
        state = self.state
        type_filter = self.combo_type.currentText()
        model_filter = self.combo_model.currentData()
        rows = []
        iou_tp = state.params.iou_tp

        for mi, rs in state.results.items():
            if model_filter is not None and model_filter != -1 and mi != model_filter:
                continue
            alias = state.model_slots[mi].alias
            for r in rs:
                if not r.has_gt: continue
                m = match_image(r.predictions, r.gt, iou_thr=iou_tp)
                # FP
                if type_filter in ("Todos", "FP"):
                    for pi in m.fp_idx:
                        d = r.predictions[pi]
                        rows.append((alias, r.image_path, "FP", "—", d.class_name, f"{d.conf:.2f}"))
                # FN
                if type_filter in ("Todos", "FN"):
                    for gi in m.fn_idx:
                        d = r.gt[gi]
                        rows.append((alias, r.image_path, "FN", d.class_name, "—", "—"))
                # MISCLS
                if type_filter in ("Todos", "MISCLS"):
                    for pi, gi in m.miscls_pairs:
                        rows.append((
                            alias, r.image_path, "MISCLS",
                            r.gt[gi].class_name, r.predictions[pi].class_name,
                            f"{r.predictions[pi].conf:.2f}"
                        ))

        self.lbl_count.setText(f"{len(rows)} error{'es' if len(rows)!=1 else ''}")
        self.table.setRowCount(len(rows))
        badge_colors = {"FP": T.WARN, "FN": T.ERR, "MISCLS": T.VIO}
        for i, (alias, p, tp_kind, cgt, cpr, cf) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(alias))
            self.table.setItem(i, 1, QTableWidgetItem(p.name))
            it = QTableWidgetItem(tp_kind)
            it.setTextAlignment(Qt.AlignCenter)
            from PySide6.QtGui import QBrush, QColor
            it.setForeground(QBrush(QColor(badge_colors.get(tp_kind, T.INK))))
            self.table.setItem(i, 2, it)
            self.table.setItem(i, 3, QTableWidgetItem(cgt))
            self.table.setItem(i, 4, QTableWidgetItem(cpr))
            self.table.setItem(i, 5, QTableWidgetItem(cf))
            self.table.item(i, 0).setData(Qt.UserRole, alias)
            self.table.item(i, 1).setData(Qt.UserRole, str(p))

    def _open_image(self, row: int, col: int):
        if row < 0 or self.state.run_dir is None: return
        alias = self.table.item(row, 0).data(Qt.UserRole)
        img = Path(self.table.item(row, 1).data(Qt.UserRole))
        annot = self.state.run_dir / alias / f"{img.stem}_annot.png"
        if annot.exists():
            os.startfile(str(annot))
