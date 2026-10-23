#!/bin/bash
# Poly-X · Motor opcional RF-DETR (macOS)
# Poly-X funciona sin el; solo hace falta para entrenar o usar modelos .pth.

cd "$(dirname "$0")" || exit 1
fin() { read -r -p "  Pulsa Enter para cerrar..."; exit "${1:-0}"; }

echo ""
echo "  Poly-X  ·  Motor opcional RF-DETR"
echo ""

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "  [ERROR] No encuentro .venv. Ejecuta primero Lanzar_macOS.command."
    fin 1
fi

echo "  Se instalan unos 50 paquetes (varios cientos de MB)."
echo "  numpy, OpenCV y Ultralytics quedan en las versiones de Poly-X."
echo ""
if ! "$VENV_PY" -m pip install -r requirements-rfdetr.txt; then
    echo "  [ERROR] La instalacion fallo. Revisa el mensaje de arriba."
    fin 1
fi

# Con numpy 2 el entrenamiento de Ultralytics 8.3.40 muere al validar.
if ! "$VENV_PY" -c "import numpy, ultralytics, rfdetr; assert numpy.__version__.startswith('1.'), 'numpy ' + numpy.__version__; print('  [OK] RF-DETR instalado - numpy', numpy.__version__)"; then
    echo "  [ERROR] La instalacion cambio numpy a 2.x. Avisa antes de entrenar."
    fin 1
fi
echo ""
echo "  Listo. Cierra y vuelve a abrir Poly-X para que aparezca RF-DETR."
fin 0
