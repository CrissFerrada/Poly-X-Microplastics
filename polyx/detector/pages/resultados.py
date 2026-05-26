"""Página 6 — Resultados. KPIs grandes + tabla por imagen."""
from __future__ import annotations
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView,
)

from ._base import DetectorPage
from ...core import theme as T
from ...core.widgets import KPICard


class ResultadosPage(DetectorPage):
    STEP_N = 6
    STEP_TITLE = "Resultados"
    STEP_DESCRIPTION = (
        "Resumen cuantitativo del análisis. Para gráficos completos y galería por imagen, "
        "genera el reporte HTML en la pestaña Reporte."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── KPIs ──
        c1, l1 = self.card("Métricas generales", "📊")
        grid = QGridLayout()
        grid.setSpacing(12)
        self.kpi_imgs   = KPICard("Imágenes", T.ACCENT)
        self.kpi_dets   = KPICard("Detecciones", T.ACCENT)
        self.kpi_conf   = KPICard("Confianza media", T.VIO)
        self.kpi_size   = KPICard("Tamaño medio (μm)", T.WARN)
        self.kpi_tp     = KPICard("TP", T.OK)
        self.kpi_fp     = KPICard("FP", T.WARN)
        self.kpi_fn     = KPICard("FN", T.ERR)
        self.kpi_f1     = KPICard("F1", T.ACCENT_D)
        grid.addWidget(self.kpi_imgs, 0, 0)
        grid.addWidget(self.kpi_dets, 0, 1)
        grid.addWidget(self.kpi_conf, 0, 2)
        grid.addWidget(self.kpi_size, 0, 3)
        grid.addWidget(self.kpi_tp, 1, 0)
        grid.addWidget(self.kpi_fp, 1, 1)
        grid.addWidget(self.kpi_fn, 1, 2)
        grid.addWidget(self.kpi_f1, 1, 3)
        l1.addLayout(grid)
        self.body.addWidget(c1)

        # ── Métricas detalladas ──
        c2, l2 = self.card("Métricas detalladas (modelo principal)", "📋")
        row = QHBoxLayout()
        row.setSpacing(20)
        self.lbl_prec = QLabel("Precision:  —")
        self.lbl_rec  = QLabel("Recall:  —")
        self.lbl_mis  = QLabel("MisCls:  —")
        for l in (self.lbl_prec, self.lbl_rec, self.lbl_mis):
            l.setStyleSheet(f"color: {T.INK2}; font-size: 10.5pt; font-weight: 500; border: none;")
            row.addWidget(l)
        row.addStretch(1)
        l2.addLayout(row)
        self.body.addWidget(c2)

        # ── Tabla por imagen ──
        c3, l3 = self.card("Por imagen", "🖼")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Modelo", "Imagen", "Pred", "GT", "TP", "FP", "FN", "Conf media"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._open_row)
        self.table.setMinimumHeight(280)
        l3.addWidget(self.table)
        self.body.addWidget(c3)

        # Suscribir
        self.state.run_finished.connect(self.refresh)
        self.state.run_aborted.connect(self.refresh)

    # ──────────────────────────────────────────
    def refresh(self):
        state = self.state
        # KPIs
        total_imgs = len({r.image_path for rs in state.results.values() for r in rs})
        all_results = [r for rs in state.results.values() for r in rs]
        total_dets = sum(len(r.predictions) for r in all_results)
        confs = [p.conf for r in all_results for p in r.predictions]
        sizes = [p.diam_um for r in all_results for p in r.predictions if p.diam_um]
        tp = sum(r.tp for r in all_results)
        fp = sum(r.fp for r in all_results)
        fn = sum(r.fn for r in all_results)

        self.kpi_imgs.set_value(total_imgs)
        self.kpi_dets.set_value(total_dets)
        self.kpi_conf.set_value(f"{(sum(confs)/len(confs)):.3f}" if confs else None)
        self.kpi_size.set_value(f"{(sum(sizes)/len(sizes)):.1f}" if sizes else None)
        self.kpi_tp.set_value(tp)
        self.kpi_fp.set_value(fp)
        self.kpi_fn.set_value(fn)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        any_gt = any(r.has_gt for r in all_results)
        self.kpi_f1.set_value(f"{f1:.3f}" if any_gt else None)
        self.lbl_prec.setText(f"Precision:  {prec:.3f}" if any_gt else "Precision:  —")
        self.lbl_rec.setText(f"Recall:  {rec:.3f}" if any_gt else "Recall:  —")
        mc = sum(r.miscls for r in all_results)
        self.lbl_mis.setText(f"MisCls:  {mc}" if any_gt else "MisCls:  —")

        # Tabla
        rows = []
        for mi, rs in state.results.items():
            slot = state.model_slots[mi]
            for r in rs:
                avg_c = sum(p.conf for p in r.predictions) / len(r.predictions) if r.predictions else 0
                rows.append((slot.alias, r.image_path, len(r.predictions),
                             len(r.gt), r.tp, r.fp, r.fn, avg_c))
        self.table.setRowCount(len(rows))
        for i, (alias, p, pred, gt, tp, fp, fn, ac) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(alias))
            self.table.setItem(i, 1, QTableWidgetItem(p.name))
            for col, v in enumerate([pred, gt, tp, fp, fn], start=2):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, it)
            self.table.setItem(i, 7, QTableWidgetItem(f"{ac:.3f}"))
            # guardamos la ruta para doble-clic
            self.table.item(i, 1).setData(Qt.UserRole, str(p))
            self.table.item(i, 0).setData(Qt.UserRole, alias)

    def _open_row(self, row: int, col: int):
        if row < 0 or self.state.run_dir is None: return
        alias = self.table.item(row, 0).data(Qt.UserRole)
        img = Path(self.table.item(row, 1).data(Qt.UserRole))
        annot = self.state.run_dir / alias / f"{img.stem}_annot.png"
        if annot.exists():
            os.startfile(str(annot))
