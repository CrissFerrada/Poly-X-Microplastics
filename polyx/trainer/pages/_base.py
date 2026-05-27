"""Página base del Entrenador (header con icono+título, no número)."""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame

from ...core import theme as T
from ..state import TrainerState


class TrainerPage(QWidget):
    """Página con header tipo manual: icono grande + título + subtítulo."""

    PAGE_ICON: str = "🎯"
    PAGE_TITLE: str = ""
    PAGE_DESCRIPTION: str = ""

    def __init__(self, state: TrainerState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setStyleSheet(f"QWidget {{ background: {T.BG_SOFT}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(38, 28, 38, 32)
        outer.setSpacing(18)

        # Header del paso (igual al manual: icono + título grande)
        head = QHBoxLayout()
        head.setSpacing(14)
        icon_lbl = QLabel(self.PAGE_ICON)
        icon_lbl.setStyleSheet("font-size: 28pt; border: none;")
        head.addWidget(icon_lbl)
        title_lbl = QLabel(self.PAGE_TITLE)
        title_lbl.setStyleSheet(f"color: {T.INK}; font-size: 22pt; font-weight: 600; border: none;")
        head.addWidget(title_lbl, 1)
        outer.addLayout(head)

        if self.PAGE_DESCRIPTION:
            desc = QLabel(self.PAGE_DESCRIPTION)
            desc.setWordWrap(True)
            desc.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none; margin-left: 52px;")
            outer.addWidget(desc)

        self.body = QVBoxLayout()
        self.body.setSpacing(16)
        outer.addLayout(self.body)
        outer.addStretch(1)

    def card(self, title: str, icon: str = "") -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {T.BG}; border: 1px solid {T.RULE}; border-radius: 8px; }}"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(20, 16, 20, 18)
        lay.setSpacing(12)
        if title:
            row = QHBoxLayout()
            row.setSpacing(8)
            if icon:
                ic = QLabel(icon)
                ic.setStyleSheet("font-size: 14pt; border: none;")
                row.addWidget(ic)
            t = QLabel(title)
            t.setStyleSheet(f"color: {T.INK}; font-size: 13pt; font-weight: 600; border: none;")
            row.addWidget(t)
            row.addStretch(1)
            lay.addLayout(row)
        return frame, lay
