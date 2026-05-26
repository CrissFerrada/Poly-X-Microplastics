"""Paleta y QSS comunes para Poly-X. Coincide con el manual HTML."""
from PySide6.QtGui import QColor

# Tinta y fondos (idéntico al :root del Manual_PolyX.html)
INK        = "#1f2328"
INK2       = "#424a53"
INK3       = "#656d76"
MUTED      = "#8c959f"
RULE       = "#d0d7de"
RULE_SOFT  = "#eaeef2"
BG         = "#ffffff"
BG_SOFT    = "#f6f8fa"

ACCENT     = "#0969da"
ACCENT_D   = "#0550ae"

OK         = "#1f6b5e"
WARN       = "#9a6700"
ERR        = "#cf222e"
VIO        = "#6639ba"

# Colores de polímero (Nile Red bajo UV 254 nm)
CLASS_COLOR_HEX = {
    "PET":  "#e3342f",   # rojo
    "PP":   "#ff8c00",   # naranjo
    "LDPE": "#ffd700",   # amarillo
    "PE":   "#ffd700",   # alias
}
def class_qcolor(name: str) -> QColor:
    return QColor(CLASS_COLOR_HEX.get(name, "#888888"))


APP_FONT_FAMILY = "Segoe UI"

GLOBAL_QSS = f"""
* {{
    font-family: "{APP_FONT_FAMILY}", Helvetica, Arial, sans-serif;
    color: {INK};
}}
QWidget {{
    background: {BG};
}}

/* Botones primarios */
QPushButton#primary {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 10pt;
}}
QPushButton#primary:hover {{
    background: {ACCENT_D};
}}
QPushButton#primary:disabled {{
    background: {RULE};
    color: {MUTED};
}}

/* Botones de peligro (Detener, eliminar) */
QPushButton#danger {{
    background: {ERR};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}}
QPushButton#danger:hover {{
    background: #a40e26;
}}
QPushButton#danger:disabled {{
    background: {RULE_SOFT};
    color: {MUTED};
}}

/* Botones secundarios (default) */
QPushButton {{
    background: {BG};
    color: {INK2};
    border: 1px solid {RULE};
    border-radius: 6px;
    padding: 7px 14px;
}}
QPushButton:hover {{
    background: {BG_SOFT};
    border-color: {MUTED};
}}
QPushButton:disabled {{
    color: {MUTED};
    background: {BG_SOFT};
}}

/* Inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG};
    color: {INK};
    border: 1px solid {RULE};
    border-radius: 5px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}

/* Tarjetas (QGroupBox) */
QGroupBox {{
    background: {BG};
    border: 1px solid {RULE};
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 16px 16px 16px;
    font-weight: 600;
    color: {INK};
    font-size: 11pt;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 12px;
    background: {BG};
}}

/* Tablas */
QTableWidget, QTableView {{
    background: {BG};
    alternate-background-color: {BG_SOFT};
    gridline-color: {RULE_SOFT};
    border: 1px solid {RULE};
    border-radius: 6px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QHeaderView::section {{
    background: {BG_SOFT};
    color: {INK2};
    padding: 8px 10px;
    border: none;
    border-right: 1px solid {RULE_SOFT};
    border-bottom: 1px solid {RULE};
    font-weight: 600;
}}

/* Barras de progreso */
QProgressBar {{
    background: {BG_SOFT};
    border: 1px solid {RULE};
    border-radius: 6px;
    text-align: center;
    color: {INK2};
    height: 18px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 5px;
}}

/* Scrollbars discretas */
QScrollBar:vertical {{
    background: {BG_SOFT};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {RULE};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {BG_SOFT};
    height: 10px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {RULE};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    height: 0px;
    width: 0px;
}}

/* Tooltips */
QToolTip {{
    background: {INK};
    color: white;
    border: none;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 9pt;
}}

/* Labels especiales */
QLabel[role="kicker"] {{
    color: {ACCENT_D};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}}
QLabel[role="h1"] {{
    color: {INK};
    font-size: 28pt;
    font-weight: 600;
}}
QLabel[role="h2"] {{
    color: {INK};
    font-size: 18pt;
    font-weight: 600;
}}
QLabel[role="h3"] {{
    color: {INK2};
    font-size: 13pt;
    font-weight: 600;
}}
QLabel[role="muted"] {{
    color: {INK3};
    font-size: 9.5pt;
}}
QLabel[role="caption"] {{
    color: {INK3};
    font-size: 9pt;
}}
"""
