"""Permite `python -m polyx.trainer`."""
import sys
from PySide6.QtWidgets import QApplication

from .window import TrainerWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = TrainerWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
