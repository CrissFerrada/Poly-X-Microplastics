"""Página 9 — Reporte HTML paper-quality."""
from __future__ import annotations
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QCheckBox, QMessageBox,
    QFileDialog, QApplication, QRadioButton, QButtonGroup, QListWidget,
    QListWidgetItem,
)

from ._base import DetectorPage
from ...core import theme as T
from ...core import pdf_export
from ...core.report_html import generate_report
from ...core.i18n import tr


class ReportePage(DetectorPage):
    STEP_N = 9
    STEP_TITLE = tr("Reporte paper-quality")
    STEP_DESCRIPTION = (
        tr("Genera un informe HTML autocontenido (todas las imágenes embebidas en base64, "
        "así no se rompen al enviarlo a otra persona) con métodos, métricas, gráficos, "
        "galería comparativa y análisis de errores. Puedes exportarlo directamente a PDF "
        "para enviarlo.")
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Tarjeta: Generar informe ──
        c1, l1 = self.card(tr("Generar informe"), "📄")
        info = QLabel(
            tr("Después de ejecutar al menos una detección, presiona <b>Generar reporte HTML</b> "
            "(se crea <code>reporte_paper.html</code> dentro de la carpeta del run y se abre "
            "en tu navegador) o <b>Exportar a PDF</b> para obtener un archivo listo para enviar. "
            "Todas las imágenes van embebidas en base64, por lo que el archivo es autocontenido.")
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l1.addWidget(info)

        # Opciones
        self.chk_refs = QCheckBox(tr("Incluir referencias bibliográficas del autor en el reporte"))
        self.chk_refs.setChecked(True)
        self.chk_refs.setStyleSheet(f"color: {T.INK2}; border: none;")
        l1.addWidget(self.chk_refs)

        self.chk_gallery = QCheckBox(tr("Incluir galería comparativa (Predicción vs Ground Truth)"))
        self.chk_gallery.setChecked(True)
        self.chk_gallery.setStyleSheet(f"color: {T.INK2}; border: none;")
        l1.addWidget(self.chk_gallery)

        # ── Alcance del informe ──
        self.body.addWidget(self._card_alcance())

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_gen = QPushButton(tr("📄  Generar reporte HTML"))
        self.btn_gen.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_gen.setCursor(Qt.PointingHandCursor)
        self.btn_gen.clicked.connect(self._generate)
        row.addWidget(self.btn_gen)

        self.btn_pdf = QPushButton(tr("📑  Exportar a PDF"))
        self.btn_pdf.setStyleSheet(
            f"background: {T.OK}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        self.btn_pdf.setCursor(Qt.PointingHandCursor)
        self.btn_pdf.setToolTip(tr("Genera el reporte y lo guarda como PDF listo para enviar."))
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
        c2, l2 = self.card(tr("Contenido del reporte"), "📋")
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

    # ── Alcance ─────────────────────────────────────────────────
    def _card_alcance(self):
        """Tarjeta para elegir si el informe cubre todo el trabajo o unas fotos."""
        c, l = self.card(tr("Alcance del informe"), "🎯")

        ayuda = QLabel(tr(
            "Puedes generar el informe del <b>trabajo completo</b>, solo de las "
            "<b>fotos que elijas</b>, o ambos de una vez. Las cifras, los gráficos "
            "y la matriz de confusión se recalculan sobre lo que elijas, así que "
            "el informe siempre describe exactamente las fotos que muestra."))
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l.addWidget(ayuda)

        self.rb_completo = QRadioButton(tr("Trabajo completo (todas las fotos analizadas)"))
        self.rb_elegidas = QRadioButton(tr("Solo las fotos que marque abajo"))
        self.rb_ambos = QRadioButton(tr("Ambos: un informe completo y otro con las marcadas"))
        self.rb_completo.setChecked(True)
        self._grupo_alcance = QButtonGroup(self)
        for rb in (self.rb_completo, self.rb_elegidas, self.rb_ambos):
            rb.setStyleSheet(f"color: {T.INK2}; border: none;")
            self._grupo_alcance.addButton(rb)
            rb.toggled.connect(self._sync_alcance)
            l.addWidget(rb)

        fila = QHBoxLayout()
        self.btn_todas = QPushButton(tr("Marcar todas"))
        self.btn_ninguna = QPushButton(tr("Desmarcar todas"))
        self.btn_todas.clicked.connect(lambda: self._marcar_todas(True))
        self.btn_ninguna.clicked.connect(lambda: self._marcar_todas(False))
        for b in (self.btn_todas, self.btn_ninguna):
            b.setCursor(Qt.PointingHandCursor)
            fila.addWidget(b)
        fila.addStretch(1)
        self.lbl_sel = QLabel("")
        self.lbl_sel.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        fila.addWidget(self.lbl_sel)
        l.addLayout(fila)

        self.lst_fotos = QListWidget()
        self.lst_fotos.setMaximumHeight(220)
        self.lst_fotos.itemChanged.connect(self._sync_alcance)
        l.addWidget(self.lst_fotos)

        self._sync_alcance()
        return c

    def refresh(self):
        """Rellena la lista con las fotos que realmente se analizaron."""
        super_refresh = getattr(super(), "refresh", None)
        if callable(super_refresh):
            super_refresh()
        rutas = sorted({r.image_path for rs in self.state.results.values() for r in rs},
                       key=lambda p: p.name)
        previas = self._fotos_marcadas()
        self.lst_fotos.blockSignals(True)
        self.lst_fotos.clear()
        for p in rutas:
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, str(p))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if p in previas else Qt.Unchecked)
            self.lst_fotos.addItem(it)
        self.lst_fotos.blockSignals(False)
        self._sync_alcance()

    def _marcar_todas(self, marcar: bool):
        self.lst_fotos.blockSignals(True)
        for i in range(self.lst_fotos.count()):
            self.lst_fotos.item(i).setCheckState(Qt.Checked if marcar else Qt.Unchecked)
        self.lst_fotos.blockSignals(False)
        self._sync_alcance()

    def _fotos_marcadas(self) -> set:
        marcadas = set()
        for i in range(self.lst_fotos.count()):
            it = self.lst_fotos.item(i)
            if it.checkState() == Qt.Checked:
                marcadas.add(Path(it.data(Qt.UserRole)))
        return marcadas

    def _sync_alcance(self):
        """Habilita la lista solo cuando el alcance la usa, y avisa si esta vacia."""
        usa_lista = self.rb_elegidas.isChecked() or self.rb_ambos.isChecked()
        for w in (self.lst_fotos, self.btn_todas, self.btn_ninguna):
            w.setEnabled(usa_lista)
        n = len(self._fotos_marcadas())
        total = self.lst_fotos.count()
        if not usa_lista:
            self.lbl_sel.setText(tr(f"{total} foto(s) en el trabajo"))
        elif n == 0:
            self.lbl_sel.setText(tr("⚠ ninguna marcada"))
        else:
            self.lbl_sel.setText(tr(f"{n} de {total} marcadas"))

    def _check_results(self) -> bool:
        """Avisa si todavía no hay detecciones ejecutadas."""
        if not self.state.results:
            QMessageBox.warning(
                self, tr("Sin resultados"),
                tr("Aún no has ejecutado ninguna detección.\n"
                "Ve a la pestaña Ejecutar e inicia el análisis primero.")
            )
            return False
        return True

    def _write_html(self, out_path: Path, solo_imagenes=None) -> Path:
        """Genera el HTML autocontenido (imágenes en base64) en `out_path`."""
        generate_report(
            self.state, out_path,
            include_refs=self.chk_refs.isChecked(),
            include_gallery=self.chk_gallery.isChecked(),
            solo_imagenes=solo_imagenes,
        )
        return out_path

    def _trabajos(self) -> list:
        """Informes a generar según el alcance: (sufijo, fotos, etiqueta).

        ``fotos`` None significa el trabajo completo. Devuelve lista vacía si el
        alcance pide fotos marcadas y no hay ninguna, para no generar en silencio
        un informe vacío que parecería decir que no se detectó nada.
        """
        marcadas = self._fotos_marcadas()
        pide_marcadas = self.rb_elegidas.isChecked() or self.rb_ambos.isChecked()
        if pide_marcadas and not marcadas:
            QMessageBox.warning(
                self, tr("Sin fotos marcadas"),
                tr("Elegiste generar el informe de fotos concretas, pero no hay "
                   "ninguna marcada.\n\nMarca al menos una en la lista, o cambia "
                   "el alcance a 'Trabajo completo'."))
            return []

        trabajos = []
        if self.rb_completo.isChecked() or self.rb_ambos.isChecked():
            trabajos.append(("", None, tr("trabajo completo")))
        if pide_marcadas:
            trabajos.append(("_seleccion", marcadas,
                             tr(f"{len(marcadas)} foto(s) marcada(s)")))
        return trabajos

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

        trabajos = self._trabajos()
        if not trabajos:
            return

        self.btn_gen.setEnabled(False)
        self.lbl_status.setText(tr("Generando reporte… (puede tardar unos segundos)"))
        generados = []
        try:
            for sufijo, fotos, etiqueta in trabajos:
                out_path = out_dir / f"reporte_paper{sufijo}.html"
                self._write_html(out_path, solo_imagenes=fotos)
                generados.append(out_path)
            self.lbl_status.setText(
                "✓ " + tr(f"{len(generados)} informe(s) generado(s) en {out_dir}"))
            for p in generados:
                try:
                    os.startfile(str(p))
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, tr("Error generando reporte"), f"{type(e).__name__}: {e}")
            self.lbl_status.setText(tr("✗ Falló la generación."))
        finally:
            self.btn_gen.setEnabled(True)

    def _export_pdf(self):
        if not self._check_results():
            return

        if not pdf_export.is_available():
            QMessageBox.warning(
                self, tr("PDF no disponible"),
                tr("Para exportar a PDF se necesita QtWebEngine (incluido en PySide6-Addons).\n\n"
                "Alternativa: pulsa 'Generar reporte HTML' y, en el navegador, usa\n"
                "Ctrl+P → 'Guardar como PDF'.")
            )
            return

        trabajos = self._trabajos()
        if not trabajos:
            return

        # Carpeta del run como base; si no existe, junto al PDF elegido
        base_dir = self.state.run_dir if (self.state.run_dir and self.state.run_dir.exists()) else Path.home()
        suggested = str(base_dir / "reporte_paper.pdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar reporte como PDF", suggested, "PDF (*.pdf)"
        )
        if not path:
            return
        pdf_base = Path(path)
        html_dir = self.state.run_dir if (self.state.run_dir and self.state.run_dir.exists()) else pdf_base.parent

        self.btn_pdf.setEnabled(False)
        self.btn_gen.setEnabled(False)
        self.lbl_status.setText(tr("Generando PDF… (renderizando con el motor del navegador, unos segundos)"))
        QApplication.processEvents()
        try:
            hechos, fallidos = [], []
            for sufijo, fotos, etiqueta in trabajos:
                html_path = html_dir / f"reporte_paper{sufijo}.html"
                # Con "ambos" se escriben dos PDF: el nombre elegido por el
                # usuario para el completo, y ese mismo con sufijo para la
                # seleccion. Reutilizar el nombre pisaria el primero.
                pdf_path = (pdf_base if not sufijo
                            else pdf_base.with_name(pdf_base.stem + sufijo + ".pdf"))
                self._write_html(html_path, solo_imagenes=fotos)
                if pdf_export.html_to_pdf(html_path, pdf_path):
                    hechos.append(pdf_path)
                else:
                    fallidos.append(etiqueta)

            if hechos:
                self.lbl_status.setText(
                    "✓ " + tr(f"{len(hechos)} PDF generado(s) en {pdf_base.parent}"))
                for p in hechos:
                    try:
                        os.startfile(str(p))
                    except Exception:
                        pass
            if fallidos:
                self.lbl_status.setText(tr("✗ No se pudo generar todo el PDF."))
                QMessageBox.warning(
                    self, tr("No se pudo generar el PDF"),
                    tr("Falló el renderizado de: " + ", ".join(fallidos) +
                       ".\n\nComo alternativa, genera el HTML y usa\n"
                       "Ctrl+P → 'Guardar como PDF' en tu navegador.")
                )
        except Exception as e:
            QMessageBox.critical(self, tr("Error generando PDF"), f"{type(e).__name__}: {e}")
            self.lbl_status.setText(tr("✗ Falló la generación de PDF."))
        finally:
            self.btn_pdf.setEnabled(True)
            self.btn_gen.setEnabled(True)
