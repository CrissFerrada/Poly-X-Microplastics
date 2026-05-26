"""Página 9 — Reporte HTML paper-quality."""
from __future__ import annotations
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QCheckBox, QMessageBox,
    QFileDialog,
)

from ._base import DetectorPage
from ...core import theme as T
from ...core.report_html import generate_report


class ReportePage(DetectorPage):
    STEP_N = 9
    STEP_TITLE = "Reporte paper-quality"
    STEP_DESCRIPTION = (
        "Genera un informe HTML autocontenido (imágenes embebidas en base64) con métodos, "
        "métricas, gráficos, galería por imagen y análisis de errores. Listo para convertir "
        "a PDF desde el navegador (Ctrl+P)."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Tarjeta: Generar informe ──
        c1, l1 = self.card("Generar informe", "📄")
        info = QLabel(
            "Después de ejecutar al menos una detección, presiona 'Generar reporte HTML'. "
            "Se creará <code>reporte_paper.html</code> dentro de la carpeta del run y "
            "se abrirá automáticamente en tu navegador."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l1.addWidget(info)

        # Opciones
        self.chk_refs = QCheckBox("Incluir referencias bibliográficas del autor en el reporte")
        self.chk_refs.setChecked(True)
        self.chk_refs.setStyleSheet(f"color: {T.INK2}; border: none;")
        l1.addWidget(self.chk_refs)

        self.chk_gallery = QCheckBox("Incluir galería de imágenes anotadas")
        self.chk_gallery.setChecked(True)
        self.chk_gallery.setStyleSheet(f"color: {T.INK2}; border: none;")
        l1.addWidget(self.chk_gallery)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_gen = QPushButton("📄  Generar reporte HTML")
        self.btn_gen.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_gen.setCursor(Qt.PointingHandCursor)
        self.btn_gen.clicked.connect(self._generate)
        row.addWidget(self.btn_gen)

        self.btn_open = QPushButton("📂  Abrir results.json de otro run…")
        self.btn_open.setEnabled(False)  # placeholder
        row.addWidget(self.btn_open)
        row.addStretch(1)
        l1.addLayout(row)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color: {T.INK3}; font-size: 9.5pt; border: none;")
        l1.addWidget(self.lbl_status)
        self.body.addWidget(c1)

        # ── Contenido del reporte ──
        c2, l2 = self.card("Contenido del reporte", "📋")
        contents = QLabel(
            "<ol>"
            "<li><b>Resumen (abstract)</b> con detecciones, conf. media y tamaño medio.</li>"
            "<li><b>Métodos</b> — modelo, parámetros, calibración, dispositivo, fecha.</li>"
            "<li><b>Resultados generales</b>"
            "<ul>"
            "<li>Distribución de clases (predicciones y/o GT)</li>"
            "<li>Histograma de confianza</li>"
            "<li>Distribución de tamaños (μm) si hay calibración</li>"
            "<li>Tabla por clase con P/R/F1</li>"
            "</ul></li>"
            "<li><b>Resumen por modelo</b> (tabla comparativa)</li>"
            "<li><b>Análisis de errores</b> (solo si hay GT)"
            "<ul>"
            "<li>Matriz de confusión</li>"
            "<li>P/R/F1 por clase</li>"
            "<li>Galería de imágenes con más errores</li>"
            "</ul></li>"
            "<li><b>Galería</b> por imagen anotada (todas)</li>"
            "<li><b>Referencias bibliográficas</b></li>"
            "</ol>"
        )
        contents.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none;")
        contents.setTextFormat(Qt.RichText)
        l2.addWidget(contents)
        self.body.addWidget(c2)

    def _generate(self):
        state = self.state
        if not state.results:
            QMessageBox.warning(
                self, "Sin resultados",
                "Aún no has ejecutado ninguna detección.\n"
                "Ve a la pestaña Ejecutar e inicia el análisis primero."
            )
            return

        out_dir = state.run_dir
        if out_dir is None or not out_dir.exists():
            # Pedir carpeta destino
            d = QFileDialog.getExistingDirectory(self, "Carpeta destino del reporte")
            if not d:
                return
            out_dir = Path(d)

        out_path = out_dir / "reporte_paper.html"
        self.btn_gen.setEnabled(False)
        self.lbl_status.setText("Generando reporte… (puede tardar unos segundos)")
        try:
            generate_report(
                state, out_path,
                include_refs=self.chk_refs.isChecked(),
                include_gallery=self.chk_gallery.isChecked(),
            )
            self.lbl_status.setText(f"✓ Reporte generado: {out_path}")
            try:
                os.startfile(str(out_path))
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Error generando reporte", f"{type(e).__name__}: {e}")
            self.lbl_status.setText("✗ Falló la generación.")
        finally:
            self.btn_gen.setEnabled(True)
