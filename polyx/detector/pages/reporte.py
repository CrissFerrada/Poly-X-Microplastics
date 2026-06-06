"""Página 9 — Reporte HTML paper-quality."""
from __future__ import annotations
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QCheckBox, QMessageBox,
    QFileDialog, QApplication,
)

from ._base import DetectorPage
from ...core import theme as T
from ...core import pdf_export
from ...core.report_html import generate_report


class ReportePage(DetectorPage):
    STEP_N = 9
    STEP_TITLE = "Reporte paper-quality"
    STEP_DESCRIPTION = (
        "Genera un informe HTML autocontenido (todas las imágenes embebidas en base64, "
        "así no se rompen al enviarlo a otra persona) con métodos, métricas, gráficos, "
        "galería comparativa y análisis de errores. Puedes exportarlo directamente a PDF "
        "para enviarlo."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Tarjeta: Generar informe ──
        c1, l1 = self.card("Generar informe", "📄")
        info = QLabel(
            "Después de ejecutar al menos una detección, presiona <b>Generar reporte HTML</b> "
            "(se crea <code>reporte_paper.html</code> dentro de la carpeta del run y se abre "
            "en tu navegador) o <b>Exportar a PDF</b> para obtener un archivo listo para enviar. "
            "Todas las imágenes van embebidas en base64, por lo que el archivo es autocontenido."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l1.addWidget(info)

        # Opciones
        self.chk_refs = QCheckBox("Incluir referencias bibliográficas del autor en el reporte")
        self.chk_refs.setChecked(True)
        self.chk_refs.setStyleSheet(f"color: {T.INK2}; border: none;")
        l1.addWidget(self.chk_refs)

        self.chk_gallery = QCheckBox("Incluir galería comparativa (Predicción vs Ground Truth)")
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

        self.btn_pdf = QPushButton("📑  Exportar a PDF")
        self.btn_pdf.setStyleSheet(
            f"background: {T.OK}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_pdf.setCursor(Qt.PointingHandCursor)
        self.btn_pdf.setToolTip("Genera el reporte y lo guarda como PDF listo para enviar.")
        self.btn_pdf.clicked.connect(self._export_pdf)
        row.addWidget(self.btn_pdf)

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
            "<li>Precisión / Recall / F1 por clase</li>"
            "<li>Galería de imágenes con más errores</li>"
            "</ul></li>"
            "<li><b>Galería comparativa</b> Predicción vs Ground Truth (lado a lado)</li>"
            "<li><b>Referencias bibliográficas</b></li>"
            "</ol>"
        )
        contents.setStyleSheet(f"color: {T.INK2}; font-size: 10pt; border: none;")
        contents.setTextFormat(Qt.RichText)
        l2.addWidget(contents)
        self.body.addWidget(c2)

    def _check_results(self) -> bool:
        """Avisa si todavía no hay detecciones ejecutadas."""
        if not self.state.results:
            QMessageBox.warning(
                self, "Sin resultados",
                "Aún no has ejecutado ninguna detección.\n"
                "Ve a la pestaña Ejecutar e inicia el análisis primero."
            )
            return False
        return True

    def _write_html(self, out_path: Path) -> Path:
        """Genera el HTML autocontenido (imágenes en base64) en `out_path`."""
        generate_report(
            self.state, out_path,
            include_refs=self.chk_refs.isChecked(),
            include_gallery=self.chk_gallery.isChecked(),
        )
        return out_path

    def _generate(self):
        if not self._check_results():
            return

        out_dir = self.state.run_dir
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
            self._write_html(out_path)
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

    def _export_pdf(self):
        if not self._check_results():
            return

        if not pdf_export.is_available():
            QMessageBox.warning(
                self, "PDF no disponible",
                "Para exportar a PDF se necesita QtWebEngine (incluido en PySide6-Addons).\n\n"
                "Alternativa: pulsa 'Generar reporte HTML' y, en el navegador, usa\n"
                "Ctrl+P → 'Guardar como PDF'."
            )
            return

        # Carpeta del run como base; si no existe, junto al PDF elegido
        base_dir = self.state.run_dir if (self.state.run_dir and self.state.run_dir.exists()) else Path.home()
        suggested = str(base_dir / "reporte_paper.pdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar reporte como PDF", suggested, "PDF (*.pdf)"
        )
        if not path:
            return
        pdf_path = Path(path)
        html_dir = self.state.run_dir if (self.state.run_dir and self.state.run_dir.exists()) else pdf_path.parent
        html_path = html_dir / "reporte_paper.html"

        self.btn_pdf.setEnabled(False)
        self.btn_gen.setEnabled(False)
        self.lbl_status.setText("Generando PDF… (renderizando con el motor del navegador, unos segundos)")
        QApplication.processEvents()
        try:
            self._write_html(html_path)
            ok = pdf_export.html_to_pdf(html_path, pdf_path)
            if ok:
                self.lbl_status.setText(f"✓ PDF generado: {pdf_path}")
                try:
                    os.startfile(str(pdf_path))
                except Exception:
                    pass
            else:
                self.lbl_status.setText("✗ No se pudo generar el PDF.")
                QMessageBox.warning(
                    self, "No se pudo generar el PDF",
                    "Falló el renderizado. Como alternativa, genera el HTML y usa\n"
                    "Ctrl+P → 'Guardar como PDF' en tu navegador."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error generando PDF", f"{type(e).__name__}: {e}")
            self.lbl_status.setText("✗ Falló la generación de PDF.")
        finally:
            self.btn_pdf.setEnabled(True)
            self.btn_gen.setEnabled(True)
