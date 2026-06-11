"""Página 6 — Resultados. KPIs grandes + histograma de tamaños + tabla por imagen."""
from __future__ import annotations
import csv
import io
import os
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QFileDialog, QFrame,
)

from ._base import DetectorPage
from ...core import theme as T
from ...core.widgets import KPICard
from ...core.metrics import LABEL_TP, LABEL_FP, LABEL_FN, LABEL_MISCLS, match_image


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
        self.kpi_tp     = KPICard(LABEL_TP, T.OK)
        self.kpi_fp     = KPICard(LABEL_FP, T.WARN)
        self.kpi_fn     = KPICard(LABEL_FN, T.ERR)
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
        self.lbl_prec = QLabel("Precisión:  —")
        self.lbl_rec  = QLabel("Recall:  —")
        self.lbl_mis  = QLabel(f"{LABEL_MISCLS}:  —")
        for l in (self.lbl_prec, self.lbl_rec, self.lbl_mis):
            l.setStyleSheet(f"color: {T.INK2}; font-size: 10.5pt; font-weight: 500; border: none;")
            row.addWidget(l)
        row.addStretch(1)
        l2.addLayout(row)
        # Sugerencia automática del umbral de confianza óptimo (F1 máximo)
        self.lbl_sugg = QLabel("")
        self.lbl_sugg.setWordWrap(True)
        self.lbl_sugg.setStyleSheet(
            f"color: {T.OK}; font-size: 10pt; font-weight: 600; border: none;")
        l2.addWidget(self.lbl_sugg)
        self.body.addWidget(c2)

        # ── Histograma de tamaños (solo con calibración) ──
        self.c_hist, l_hist = self.card(
            "Distribución de tamaños (μm) — solo con calibración activa", "📐"
        )
        self._hist_placeholder = QLabel(
            "Configura μm/px en Parámetros para ver la distribución de tamaños."
        )
        self._hist_placeholder.setAlignment(Qt.AlignCenter)
        self._hist_placeholder.setStyleSheet(
            f"color: {T.INK3}; font-size: 10pt; border: none; padding: 24px;"
        )
        l_hist.addWidget(self._hist_placeholder)
        self._hist_canvas = None
        self.body.addWidget(self.c_hist)

        # ── Exportar CSV ──
        c_csv, l_csv = self.card("Exportar datos", "💾")
        row_csv = QHBoxLayout()
        btn_csv = QPushButton("📄  Exportar CSV de detecciones")
        btn_csv.clicked.connect(self._export_csv)
        row_csv.addWidget(btn_csv)
        row_csv.addStretch(1)
        l_csv.addLayout(row_csv)
        self.body.addWidget(c_csv)

        # ── Tabla por imagen ──
        c3, l3 = self.card("Por imagen", "🖼")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Modelo", "Imagen", "Pred", "GT", LABEL_TP, LABEL_FP, LABEL_FN, "Conf media"]
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
        self.lbl_prec.setText(f"Precisión:  {prec:.3f}" if any_gt else "Precisión:  —")
        self.lbl_rec.setText(f"Recall:  {rec:.3f}" if any_gt else "Recall:  —")
        mc = sum(r.miscls for r in all_results)
        self.lbl_mis.setText(f"{LABEL_MISCLS}:  {mc}" if any_gt else f"{LABEL_MISCLS}:  —")

        # Sugerencia de umbral óptimo
        sugg = self._suggest_operating_point(all_results) if any_gt else None
        if sugg:
            t, f1s, ps, rs_ = sugg
            self.lbl_sugg.setText(
                f"💡 Punto de operación sugerido: confianza ≥ {t:.2f} → "
                f"F1 {f1s:.3f} (Precisión {ps:.3f} / Recall {rs_:.3f}). "
                f"Consejo: ejecuta con confianza baja (ej. 0.05) para explorar "
                f"todo el rango y deja que esta sugerencia elija el corte.")
        else:
            self.lbl_sugg.setText("")

        # Histograma de tamaños (solo si hay calibración)
        self._update_histogram()

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

    def _suggest_operating_point(self, all_results):
        """Barre umbrales de confianza sobre las predicciones guardadas y
        devuelve (conf, f1, precision, recall) del F1 máximo contra el GT."""
        gt_results = [r for r in all_results if r.has_gt]
        if not gt_results:
            return None
        confs = sorted({round(p.conf, 2) for r in gt_results for p in r.predictions})
        if not confs:
            return None
        step = max(1, len(confs) // 25)   # acotar a ~25 umbrales
        iou_thr = self.state.params.iou_tp
        best = None
        for t in confs[::step]:
            tp = fp = fn = 0
            for r in gt_results:
                preds = [p for p in r.predictions if p.conf >= t]
                m = match_image(preds, r.gt, iou_thr)
                tp += m.tp; fp += m.fp; fn += m.fn
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if best is None or f1 > best[1]:
                best = (t, f1, prec, rec)
        return best

    def _update_histogram(self):
        """Dibuja histograma de diam_um por clase; solo si hay calibración."""
        um_per_px = getattr(self.state.params, "um_per_px", 0.0)
        all_results = [r for rs in self.state.results.values() for r in rs]
        sizes_by_class: dict[str, list[float]] = {}
        for r in all_results:
            for p in r.predictions:
                if p.diam_um and p.diam_um > 0:
                    sizes_by_class.setdefault(p.class_name, []).append(p.diam_um)

        has_data = um_per_px > 0 and any(sizes_by_class.values())

        if not has_data:
            if self._hist_canvas:
                self._hist_canvas.setVisible(False)
            self._hist_placeholder.setVisible(True)
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            if self._hist_canvas is None:
                fig = Figure(figsize=(7, 3), tight_layout=True)
                self._hist_ax = fig.add_subplot(111)
                self._hist_canvas = FigureCanvasQTAgg(fig)
                self._hist_canvas.setMinimumHeight(220)
                # Insertar después del placeholder
                l_hist = self.c_hist.layout()
                l_hist.addWidget(self._hist_canvas)

            ax = self._hist_ax
            ax.clear()
            ax.set_facecolor("#f6f8fa")
            ax.figure.set_facecolor("#ffffff")

            colors = {"PET": "#e3342f", "PP": "#ff8c00", "LDPE": "#ffd700"}
            for cls_name, vals in sizes_by_class.items():
                color = colors.get(cls_name, "#0969da")
                ax.hist(vals, bins=20, alpha=0.72, label=f"{cls_name} (n={len(vals)})",
                        color=color, edgecolor="white", linewidth=0.5)

            ax.set_xlabel("Diámetro equivalente (μm)", fontsize=9)
            ax.set_ylabel("Frecuencia", fontsize=9)
            ax.set_title("Distribución de tamaños por clase", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(labelsize=8)
            self._hist_canvas.draw()
            self._hist_placeholder.setVisible(False)
            self._hist_canvas.setVisible(True)
        except Exception as e:
            self._hist_placeholder.setText(f"matplotlib no disponible: {e}")
            self._hist_placeholder.setVisible(True)

    def _export_csv(self):
        """Exporta todas las detecciones a un archivo CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar CSV", "detecciones.csv", "CSV (*.csv)"
        )
        if not path:
            return
        all_results = [r for rs in self.state.results.values() for r in rs]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["modelo", "imagen", "clase", "conf",
                             "x1", "y1", "x2", "y2", "diam_um"])
                for mi, rs in self.state.results.items():
                    alias = self.state.model_slots[mi].alias
                    for r in rs:
                        for p in r.predictions:
                            w.writerow([
                                alias, r.image_path.name,
                                p.class_name, f"{p.conf:.4f}",
                                f"{p.x1:.1f}", f"{p.y1:.1f}",
                                f"{p.x2:.1f}", f"{p.y2:.1f}",
                                f"{p.diam_um:.2f}" if p.diam_um else "",
                            ])
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"No se pudo guardar: {e}")

    def _open_row(self, row: int, col: int):
        if row < 0 or self.state.run_dir is None: return
        alias = self.table.item(row, 0).data(Qt.UserRole)
        img = Path(self.table.item(row, 1).data(Qt.UserRole))
        annot = self.state.run_dir / alias / f"{img.stem}_annot.png"
        if annot.exists():
            os.startfile(str(annot))
