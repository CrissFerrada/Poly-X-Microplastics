"""Punto de entrada: python -m polyx.visor"""
import sys
from PySide6.QtWidgets import QApplication
from .window import VisorWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = VisorWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
