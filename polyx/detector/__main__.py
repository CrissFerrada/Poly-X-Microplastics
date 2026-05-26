"""Permite `python -m polyx.detector`."""
import sys
from PySide6.QtWidgets import QApplication

from .window import DetectorWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = DetectorWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
