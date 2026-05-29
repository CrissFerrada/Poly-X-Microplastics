"""Regenera las capturas del Manual_PolyX.html con la UI actual.

Captura cada módulo disponible (los que ya existan en polyx/) y reemplaza
su imagen correspondiente en el manual conservando todo el texto intacto.

Uso:
    .venv\\Scripts\\python.exe generar_manual.py [--solo launcher|detector|...|todos]

El script es seguro de ejecutar muchas veces: si un módulo no existe todavía,
simplemente se salta su captura y deja la del manual original.
"""
from __future__ import annotations
import sys
import re
import base64
import argparse
import time
from pathlib import Path

# Forzar UTF-8 en la consola de Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from PySide6.QtCore import QTimer, Qt, QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication


# ────────────────────────────────────────────────────────────────────
# Mapa figura -> (módulo dotted, clase ventana, tamaño W x H, espera ms)
# ────────────────────────────────────────────────────────────────────
FIGURE_MAP = {
    # fig_01 — Launcher principal
    1:  dict(module="polyx.launcher",     cls="LauncherWindow",  size=(1180, 820), wait_ms=900),

    # fig_02..09 — Detector (9 pestañas)
    2:  dict(module="polyx.detector",     cls="DetectorWindow",  size=(1280, 820), tab=0),  # Modelos
    3:  dict(module="polyx.detector",     cls="DetectorWindow",  size=(1280, 820), tab=1),  # Imágenes
    4:  dict(module="polyx.detector",     cls="DetectorWindow",  size=(1280, 820), tab=3),  # Parámetros
    5:  dict(module="polyx.detector",     cls="DetectorWindow",  size=(1280, 820), tab=4),  # Ejecutar
    6:  dict(module="polyx.detector",     cls="DetectorWindow",  size=(1280, 820), tab=5),  # Resultados
    7:  dict(module="polyx.detector",     cls="DetectorWindow",  size=(1280, 820), tab=6),  # Errores
    8:  dict(module="polyx.detector",     cls="DetectorWindow",  size=(1280, 820), tab=7),  # Comparar
    9:  dict(module="polyx.detector",     cls="DetectorWindow",  size=(1280, 820), tab=8),  # Reporte

    # fig_10..18 — Entrenador (9 pestañas)
    10: dict(module="polyx.trainer",      cls="TrainerWindow",   size=(1320, 840), tab=0),  # Modelo
    11: dict(module="polyx.trainer",      cls="TrainerWindow",   size=(1320, 840), tab=1),  # Dataset
    12: dict(module="polyx.trainer",      cls="TrainerWindow",   size=(1320, 840), tab=2),  # Parámetros
    13: dict(module="polyx.trainer",      cls="TrainerWindow",   size=(1320, 840), tab=3),  # Augmentación
    14: dict(module="polyx.trainer",      cls="TrainerWindow",   size=(1320, 840), tab=4),  # Entrenar
    15: dict(module="polyx.trainer",      cls="TrainerWindow",   size=(1320, 840), tab=5),  # Evaluar
    16: dict(module="polyx.trainer",      cls="TrainerWindow",   size=(1320, 840), tab=6),  # Comparar
    17: dict(module="polyx.trainer",      cls="TrainerWindow",   size=(1320, 840), tab=7),  # Exportar
    18: dict(module="polyx.trainer",      cls="TrainerWindow",   size=(1320, 840), tab=8),  # Informe

    # fig_19 — Etiquetador
    19: dict(module="polyx.etiquetador",  cls="LabelerWindow",   size=(1400, 880), wait_ms=800),

    # fig_20 — Visor
    20: dict(module="polyx.visor",        cls="VisorWindow",     size=(1420, 900), wait_ms=800),
}


# ────────────────────────────────────────────────────────────────────
def pixmap_to_png_base64(pix: QPixmap) -> str:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    pix.save(buf, "PNG")
    return base64.b64encode(bytes(ba)).decode("ascii")


def capture_window(module_dotted: str, cls_name: str, size, wait_ms: int = 600,
                   tab: int | None = None) -> str | None:
    """Importa el módulo, instancia la ventana (cambia de pestaña si aplica)
    y devuelve base64 PNG. Devuelve None si el módulo no existe."""
    import importlib
    try:
        spec = importlib.util.find_spec(module_dotted)
        if spec is None:
            print(f"  [SKIP] módulo {module_dotted} aún no existe.")
            return None
        mod = importlib.import_module(module_dotted)
        klass = getattr(mod, cls_name, None)
        if klass is None:
            print(f"  [SKIP] clase {cls_name} no encontrada en {module_dotted}.")
            return None
    except Exception as e:
        print(f"  [ERR] no se pudo importar {module_dotted}: {e}")
        return None

    w = klass()
    if size:
        w.resize(*size)
    w.move(-3000, -3000)
    w.show()

    # Si pidieron una pestaña concreta (Detector u otros con sidebar)
    if tab is not None:
        try:
            btns = getattr(w, "sidebar_buttons", None)
            stack = getattr(w, "stack", None)
            if btns and stack and 0 <= tab < len(btns):
                btns[tab].setChecked(True)
                stack.setCurrentIndex(tab)
        except Exception as e:
            print(f"  [WARN] no se pudo cambiar a pestaña {tab}: {e}")

    deadline = time.time() + wait_ms / 1000.0
    while time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.02)

    pix = w.grab()
    w.close()
    return pixmap_to_png_base64(pix)


# ────────────────────────────────────────────────────────────────────
def patch_manual(manual_path: Path, replacements: dict[int, str]) -> int:
    """Reemplaza el src base64 de las imágenes Nº indicadas (1-based).

    Devuelve la cantidad reemplazada.
    """
    html = manual_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"""(<img[^>]*src=['"])data:image/\w+;base64,[^'"]+(['"][^>]*>)""",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(html))
    if not matches:
        print("  [WARN] no se encontraron <img> con base64 en el manual.")
        return 0

    # Reemplazar en orden inverso para no invalidar offsets previos
    new_html = html
    done = 0
    for idx in sorted(replacements.keys(), reverse=True):
        i = idx - 1
        if i < 0 or i >= len(matches):
            print(f"  [WARN] fig_{idx:02d} fuera de rango (manual tiene {len(matches)} figuras).")
            continue
        m = matches[i]
        b64 = replacements[idx]
        new_block = f"{m.group(1)}data:image/png;base64,{b64}{m.group(2)}"
        new_html = new_html[: m.start()] + new_block + new_html[m.end():]
        done += 1
        print(f"  [OK]  fig_{idx:02d} reemplazada ({len(b64)//1024} KB base64)")

    if done:
        manual_path.write_text(new_html, encoding="utf-8")
    return done


# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Regenera capturas del Manual_PolyX.html")
    parser.add_argument("--solo", default="todos",
                        help="launcher | detector | trainer | etiquetador | visor | todos")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    manual = root / "Manual_PolyX.html"
    if not manual.exists():
        print(f"[ERROR] no se encuentra {manual}")
        return 1

    # Selección de figuras según --solo
    name_to_figs = {
        "launcher":    [1],
        "detector":    [2, 3, 4, 5, 6, 7, 8, 9],
        "trainer":     [10, 11, 12, 13, 14, 15, 16, 17, 18],
        "etiquetador": [19],
        "visor":       [20],
    }
    if args.solo == "todos":
        target_figs = set().union(*name_to_figs.values())
    else:
        target_figs = set(name_to_figs.get(args.solo, []))
        if not target_figs:
            print(f"[ERROR] opción desconocida: {args.solo}")
            return 2

    app = QApplication.instance() or QApplication(sys.argv)

    print(f"Capturando módulos UI (solo='{args.solo}')...")
    replacements: dict[int, str] = {}
    for fig_n, cfg in FIGURE_MAP.items():
        if fig_n not in target_figs:
            continue
        print(f"  → fig_{fig_n:02d}  {cfg['module']}.{cfg['cls']}")
        b64 = capture_window(
            cfg["module"], cfg["cls"],
            size=cfg.get("size"),
            wait_ms=cfg.get("wait_ms", 600),
            tab=cfg.get("tab"),
        )
        if b64:
            replacements[fig_n] = b64

    if not replacements:
        print("Nada para actualizar. (¿Aún no están implementados los módulos?)")
        return 0

    print(f"\nActualizando {manual.name}...")
    done = patch_manual(manual, replacements)
    print(f"\n✓ Manual actualizado: {done} figura(s) reemplazadas de {len(replacements)} capturadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
