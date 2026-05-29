"""Punto de entrada: python -m polyx.etiquetador"""
import sys
from PySide6.QtWidgets import QApplication
from .window import LabelerWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = LabelerWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
