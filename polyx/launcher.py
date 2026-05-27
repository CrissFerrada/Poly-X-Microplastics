"""Módulo 1 — Launcher (menú principal). Reproduce la Figura 1 del manual.

Hero con kicker, título 'Poly-X analytics', subtítulo, 4 KPI chips,
y 4 tarjetas grandes de módulo (Detector, Entrenador, Etiquetador, Visor).
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QTimer, QRectF, QPointF, QRect
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QLinearGradient, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QFrame, QScrollArea, QSpacerItem, QSizePolicy
)

from .core import theme as T
from .core.widgets import LogoBadge, HLine
from . import __version__


# ──────────────────────────────────────────────────────────────────────
# Panel decorativo: microscopio circular con partículas PET/PP/LDPE flotando
# ──────────────────────────────────────────────────────────────────────
class MicroscopePanel(QWidget):
    """Círculo oscuro con partículas de polímero flotando + barra de escala 10 μm."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 220)
        self.setMaximumWidth(360)
        import random
        rnd = random.Random(7)
        # (cx_rel, cy_rel, radius, color, kind: 0=disco, 1=fibra, 2=mancha)
        self.particles = []
        for _ in range(11):
            cx = rnd.uniform(0.15, 0.85)
            cy = rnd.uniform(0.15, 0.85)
            r = rnd.uniform(6, 18)
            color = rnd.choice([T.CLASS_COLOR_HEX["PET"],
                                T.CLASS_COLOR_HEX["PP"],
                                T.CLASS_COLOR_HEX["LDPE"]])
            kind = rnd.choices([0, 1, 2], weights=[5, 3, 2])[0]
            angle = rnd.uniform(0, 360)
            self.particles.append((cx, cy, r, color, kind, angle))
        self._phase = 0.0
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(60)

    def _tick(self):
        self._phase = (self._phase + 0.04) % (2 * 3.14159)
        self.update()

    def paintEvent(self, e):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        side = min(W, H) - 14
        cx, cy = W / 2, H / 2

        # Anillo exterior del objetivo
        ring_rect = QRectF(cx - side/2 - 2, cy - side/2 - 2, side + 4, side + 4)
        p.setPen(QPen(QColor("#22272e"), 4))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(ring_rect)

        # Fondo del campo (gradiente oscuro fluorescente)
        field_rect = QRectF(cx - side/2, cy - side/2, side, side)
        grad = QLinearGradient(field_rect.topLeft(), field_rect.bottomRight())
        grad.setColorAt(0.0, QColor("#0d1117"))
        grad.setColorAt(1.0, QColor("#1c2128"))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(field_rect)

        # Clip al círculo
        p.save()
        path_rect = field_rect
        p.setClipRect(self.rect())
        from PySide6.QtGui import QPainterPath
        clip = QPainterPath()
        clip.addEllipse(path_rect)
        p.setClipPath(clip)

        # Partículas
        for i, (rx, ry, r, color, kind, angle) in enumerate(self.particles):
            off = 4 * math.sin(self._phase + i)
            x = field_rect.left() + rx * side
            y = field_rect.top() + ry * side + off
            c = QColor(color)
            # halo (fluorescencia)
            halo = QColor(c); halo.setAlpha(80)
            p.setBrush(QBrush(halo)); p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x, y), r * 1.8, r * 1.8)
            # cuerpo
            p.setBrush(QBrush(c))
            if kind == 0:  # disco
                p.drawEllipse(QPointF(x, y), r, r)
            elif kind == 1:  # fibra (rectángulo rotado)
                p.save()
                p.translate(x, y)
                p.rotate(angle + 15 * math.sin(self._phase + i))
                p.drawRoundedRect(QRectF(-r * 1.6, -r * 0.25, r * 3.2, r * 0.5), 2, 2)
                p.restore()
            else:  # mancha
                pp = QPainterPath()
                pp.addEllipse(QPointF(x, y), r * 1.1, r * 0.7)
                p.drawPath(pp)

        p.restore()

        # Barra de escala 10 μm
        bar_w = side * 0.18
        bar_y = cy + side / 2 - 14
        bar_x = cx + side / 2 - bar_w - 14
        p.setPen(QPen(QColor("white"), 2))
        p.drawLine(QPointF(bar_x, bar_y), QPointF(bar_x + bar_w, bar_y))
        p.drawLine(QPointF(bar_x, bar_y - 4), QPointF(bar_x, bar_y + 4))
        p.drawLine(QPointF(bar_x + bar_w, bar_y - 4), QPointF(bar_x + bar_w, bar_y + 4))
        p.setFont(QFont(T.APP_FONT_FAMILY, 8, QFont.Bold))
        p.drawText(QPointF(bar_x + bar_w / 2 - 14, bar_y - 6), "10 μm")


# ──────────────────────────────────────────────────────────────────────
# Chips KPI superiores ('3 / POLÍMEROS / PET·PP·LDPE')
# ──────────────────────────────────────────────────────────────────────
class StatChip(QFrame):
    def __init__(self, big: str, label: str, sub: str, color: str):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        self.setMinimumHeight(96)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)

        l_big = QLabel(big)
        l_big.setStyleSheet(f"color: {color}; font-size: 26pt; font-weight: 700; border: none;")
        l_label = QLabel(label.upper())
        l_label.setStyleSheet(
            f"color: {T.INK2}; font-size: 8.5pt; font-weight: 700; "
            f"letter-spacing: 1.5px; border: none;"
        )
        l_sub = QLabel(sub)
        l_sub.setStyleSheet(f"color: {T.INK3}; font-size: 9pt; border: none;")
        l_sub.setWordWrap(True)

        lay.addWidget(l_big)
        lay.addWidget(l_label)
        lay.addWidget(l_sub)


# ──────────────────────────────────────────────────────────────────────
# Tarjeta de módulo
# ──────────────────────────────────────────────────────────────────────
class ModuleCard(QFrame):
    def __init__(self, number: int, icon: str, title: str, description: str,
                 accent: str, on_open):
        super().__init__()
        self._on_open = on_open
        self.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 10px; }}"
        )
        self.setMinimumHeight(170)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(8)

        top = QHBoxLayout()
        tag = QLabel(f"MÓDULO 0{number}")
        tag.setStyleSheet(
            f"color: white; background: {accent}; padding: 3px 8px; "
            f"border-radius: 4px; font-size: 8.5pt; font-weight: 700; "
            f"letter-spacing: 1.2px; border: none;"
        )
        tag.setFixedHeight(20)
        tag.setAlignment(Qt.AlignCenter)
        top.addWidget(tag, 0, Qt.AlignLeft | Qt.AlignVCenter)
        top.addStretch(1)
        btn_open = QPushButton("Abrir  →")
        btn_open.setObjectName("primary")
        btn_open.setCursor(Qt.PointingHandCursor)
        btn_open.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {accent}; "
            f"border: none; font-weight: 600; padding: 4px 0; }}"
            f"QPushButton:hover {{ color: {T.ACCENT_D}; }}"
        )
        btn_open.clicked.connect(self._on_open)
        top.addWidget(btn_open)
        lay.addLayout(top)

        title_row = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"font-size: 22pt; border: none;")
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {T.INK}; font-size: 17pt; font-weight: 600; border: none;"
        )
        title_row.addWidget(icon_lbl)
        title_row.addWidget(title_lbl)
        title_row.addStretch(1)
        lay.addLayout(title_row)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        lay.addWidget(desc)
        lay.addStretch(1)

        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._on_open()
        super().mousePressEvent(ev)


# ──────────────────────────────────────────────────────────────────────
# Ventana principal Launcher
# ──────────────────────────────────────────────────────────────────────
class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poly-X · Suite de microplásticos")
        self.resize(1180, 820)
        self.setStyleSheet(T.GLOBAL_QSS + f"QMainWindow {{ background: {T.BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Topbar ──
        topbar = QFrame()
        topbar.setStyleSheet(f"background: {T.BG}; border-bottom: 1px solid {T.RULE};")
        topbar.setFixedHeight(58)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(24, 0, 24, 0)
        tb.setSpacing(12)
        tb.addWidget(LogoBadge("POLY-X", "Microplastics analytics suite"))
        tb.addStretch(1)
        version_lbl = QLabel(f"v{__version__}")
        version_lbl.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        tb.addWidget(version_lbl)
        root.addWidget(topbar)

        # ── Scroll central ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        root.addWidget(scroll, 1)

        page = QWidget()
        scroll.setWidget(page)
        body = QVBoxLayout(page)
        body.setContentsMargins(48, 32, 48, 40)
        body.setSpacing(28)

        # ── Hero ──
        hero = QHBoxLayout()
        hero.setSpacing(36)

        left = QVBoxLayout()
        left.setSpacing(10)
        kicker = QLabel("● PLATAFORMA DE ANÁLISIS")
        kicker.setStyleSheet(
            f"color: {T.ACCENT_D}; font-size: 9pt; font-weight: 700; "
            f"letter-spacing: 1.8px; border: none;"
        )
        left.addWidget(kicker)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        t1 = QLabel("Poly-X")
        t1.setStyleSheet(f"color: {T.INK}; font-size: 38pt; font-weight: 700; border: none;")
        t2 = QLabel("analytics")
        t2.setStyleSheet(
            f"color: {T.ACCENT}; font-size: 38pt; font-weight: 300; border: none;"
        )
        title_row.addWidget(t1)
        title_row.addWidget(t2)
        title_row.addStretch(1)
        left.addLayout(title_row)

        subtitle = QLabel("Plataforma de detección y clasificación de microplásticos")
        subtitle.setStyleSheet(
            f"color: {T.INK2}; font-size: 13pt; font-weight: 500; border: none;"
        )
        left.addWidget(subtitle)

        descr = QLabel(
            "Detección automatizada de PET, PP y LDPE por fluorescencia Nile Red (254 nm) "
            "e inteligencia artificial. Entrenamiento, etiquetado, detección y reporte en un mismo flujo."
        )
        descr.setWordWrap(True)
        descr.setStyleSheet(f"color: {T.INK3}; font-size: 10.5pt; border: none;")
        left.addWidget(descr)

        # Crédito de autoría (justo bajo el descriptor)
        author = QLabel("✍  Diseñado y desarrollado por <b>Cristofher Ferrada</b> · Doctorado en Química, 2026")
        author.setTextFormat(Qt.RichText)
        author.setStyleSheet(
            f"color: {T.ACCENT_D}; font-size: 9.5pt; border: none; "
            f"background: {T.BG_SOFT}; padding: 6px 10px; border-radius: 4px;"
        )
        left.addWidget(author)
        left.addSpacing(4)

        # 4 chips
        chips = QHBoxLayout()
        chips.setSpacing(12)
        chips.addWidget(StatChip("3", "Polímeros", "PET · PP · LDPE", T.ACCENT))
        chips.addWidget(StatChip("254", "nm", "fluorescencia Nile Red", T.OK))
        chips.addWidget(StatChip("YOLO", "v8 / v11", "deep learning", T.VIO))
        chips.addWidget(StatChip("μm", "medición", "calibración por píxel", T.WARN))
        left.addLayout(chips)

        hero.addLayout(left, 3)
        hero.addWidget(MicroscopePanel(), 0, Qt.AlignTop)
        body.addLayout(hero)

        # ── Sección 'MÓDULOS' ──
        mod_kicker = QLabel("MÓDULOS")
        mod_kicker.setStyleSheet(
            f"color: {T.INK2}; font-size: 9pt; font-weight: 700; "
            f"letter-spacing: 1.8px; border: none;"
        )
        body.addWidget(mod_kicker)

        grid = QGridLayout()
        grid.setSpacing(16)
        grid.addWidget(ModuleCard(
            1, "🔬", "Detector",
            "Analiza imágenes con un modelo .pt entrenado. Genera salidas anotadas, "
            "CSV con centroides y diámetros, métricas globales y reporte HTML paper-quality.",
            T.ACCENT, self.open_detector), 0, 0)
        grid.addWidget(ModuleCard(
            2, "🎯", "Entrenador",
            "Entrena modelos YOLO v8 / v11. Curvas en vivo, recomendaciones automáticas "
            "de calidad y comparación con runs anteriores.",
            T.OK, self.open_trainer), 0, 1)
        grid.addWidget(ModuleCard(
            3, "🏷", "Etiquetador",
            "Anota imágenes en formato YOLO. Soporta pre-anotación con un modelo "
            "existente y atajos de teclado, ahorra ~80 % del tiempo manual.",
            T.WARN, self.open_labeler), 1, 0)
        grid.addWidget(ModuleCard(
            4, "📐", "Visor",
            "Inspección de una imagen a la vez con calibración interactiva μm/píxel "
            "(línea o círculo) y medición precisa por partícula.",
            T.VIO, self.open_viewer), 1, 1)
        body.addLayout(grid)

        # ── Pie ──
        body.addStretch(1)
        body.addWidget(HLine())
        foot = QHBoxLayout()
        st = QLabel("● Listo")
        st.setStyleSheet(f"color: {T.OK}; font-size: 9.5pt; font-weight: 600;")
        foot.addWidget(st)
        foot.addSpacing(20)
        copyr = QLabel("© Cristofher Ferrada · Poly-X v" + __version__)
        copyr.setStyleSheet(f"color: {T.INK3}; font-size: 9pt;")
        foot.addWidget(copyr)
        foot.addStretch(1)
        leeme_btn = QPushButton("📄 LÉAME")
        leeme_btn.clicked.connect(self.open_leeme)
        foot.addWidget(leeme_btn)
        manual_btn = QPushButton("📖 Manual de usuario")
        manual_btn.clicked.connect(self.open_manual)
        foot.addWidget(manual_btn)
        body.addLayout(foot)

    # ── Lanzadores ──────────────────────────────────────────────
    def open_detector(self): self._launch_module("polyx.detector")
    def open_trainer(self):  self._launch_module("polyx.trainer", fallback="🎯 Entrenador en construcción")
    def open_labeler(self):  self._launch_module("polyx.etiquetador", fallback="🏷 Etiquetador en construcción")
    def open_viewer(self):   self._launch_module("polyx.visor", fallback="📐 Visor en construcción")

    def _launch_module(self, dotted: str, fallback: str = ""):
        """Lanza el módulo en un proceso aparte para no bloquear el launcher."""
        try:
            # Verificar que existe sin importarlo (evita errores de Qt)
            import importlib
            spec = importlib.util.find_spec(dotted)
            if spec is None:
                self._toast(fallback or f"Módulo {dotted} no encontrado")
                return
            subprocess.Popen(
                [sys.executable, "-m", dotted],
                cwd=str(Path(__file__).resolve().parents[1]),
            )
        except Exception as e:
            self._toast(f"Error al abrir: {e}")

    def open_manual(self):
        manual = Path(__file__).resolve().parents[1] / "Manual_PolyX.html"
        if manual.exists():
            import webbrowser
            webbrowser.open(manual.as_uri())
        else:
            self._toast("Manual_PolyX.html no encontrado")

    def open_leeme(self):
        leeme = Path(__file__).resolve().parents[1] / "LEEME.txt"
        if leeme.exists():
            import os
            os.startfile(str(leeme))
        else:
            self._toast("LEEME.txt no encontrado")

    def _toast(self, msg: str):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Poly-X", msg)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = LauncherWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
