"""Elegir el checkpoint que mejor detecta en el dominio que de verdad importa.

El problema que resuelve. El dataset mezcla dos fuentes: placas dopadas de
laboratorio y sedimento real. Las dopadas aportan casi todas las cajas (son
imagenes densas), asi que dominan la validacion aunque sean pocas imagenes. En
el dataset del Loa la validacion queda con 1191 cajas de laboratorio frente a 47
de sedimento real: un 96% del dominio equivocado.

Ultralytics guarda ``best.pt`` segun el mAP de esa validacion. O sea que, sin
tocar nada, uno se lleva el checkpoint que mejor funciona sobre placas limpias de
laboratorio, que es justamente el dominio que ya se sabia que no transfiere a las
fotos de terreno.

Aqui se rehace la eleccion: se evaluan todos los checkpoints guardados contra
**solo** las imagenes de sedimento real de la validacion, y se copia el ganador
como ``best_real.pt``. No se borra ni se altera ``best.pt``: quedan los dos, y el
informe puede declarar con cual se detecto.

La separacion de dominios sale del prefijo del nombre de archivo que pone
``armar_dataset.py`` (``real__`` frente a ``lab__``). Si un dataset no trae los
dos prefijos, no hay nada que reelegir y el paso se salta.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

PREFIJO_REAL = "real__"
PREFIJO_LAB = "lab__"

# Nombre del peso reelegido. Convive con best.pt, no lo reemplaza.
NOMBRE_MEJOR_REAL = "best_real.pt"


def _imagenes_de_val(datos_yaml: Path) -> List[Path]:
    """Lista las imagenes del split de validacion declarado en el yaml."""
    cfg = yaml.safe_load(datos_yaml.read_text(encoding="utf-8")) or {}
    raiz = Path(cfg.get("path") or datos_yaml.parent)
    if not raiz.is_absolute():
        raiz = (datos_yaml.parent / raiz).resolve()

    val = cfg.get("val")
    if not val:
        return []
    ruta = Path(val)
    if not ruta.is_absolute():
        ruta = (raiz / ruta).resolve()

    if ruta.is_file():  # lista .txt de rutas
        return [Path(l.strip()) for l in ruta.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    if ruta.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return sorted(p for p in ruta.iterdir()
                      if p.is_file() and p.suffix.lower() in exts)
    return []


def dominios_presentes(datos_yaml: Path) -> Dict[str, int]:
    """Cuenta cuantas imagenes de validacion hay de cada dominio."""
    conteo = {"real": 0, "lab": 0, "sin_marcar": 0}
    for p in _imagenes_de_val(Path(datos_yaml)):
        if p.name.startswith(PREFIJO_REAL):
            conteo["real"] += 1
        elif p.name.startswith(PREFIJO_LAB):
            conteo["lab"] += 1
        else:
            conteo["sin_marcar"] += 1
    return conteo


def se_puede_reelegir(datos_yaml: Path) -> bool:
    """True si la validacion mezcla dominios y por tanto reelegir tiene sentido."""
    c = dominios_presentes(Path(datos_yaml))
    return c["real"] > 0 and c["lab"] > 0


def _etiqueta_de(imagen: Path) -> Path:
    """Ruta del .txt de una imagen, con la convencion images/ -> labels/."""
    partes = list(imagen.parts)
    for i in range(len(partes) - 1, -1, -1):
        if partes[i] == "images":
            partes[i] = "labels"
            break
    return Path(*partes).with_suffix(".txt")


def _enlazar(origen: Path, destino: Path) -> None:
    """Enlace duro si se puede; copia si no (otro volumen, sin permisos)."""
    import os
    import shutil

    if destino.exists():
        return
    try:
        os.link(origen, destino)
    except (OSError, NotImplementedError):
        shutil.copy2(origen, destino)


def crear_yaml_solo_real(datos_yaml: Path, destino_dir: Path) -> Optional[Path]:
    """Escribe un dataset.yaml cuya validacion son solo las imagenes reales.

    El subconjunto vive en su **propio arbol** images/ + labels/, no en una lista
    que apunte al split original. Ultralytics deriva la ruta del cache de
    etiquetas de la carpeta que las contiene, asi que validar un subconjunto de
    ``images/val`` reescribe ``labels/val.cache`` con solo esas imagenes: el
    siguiente entrenamiento validaria en silencio sobre el subconjunto y su
    best.pt significaria otra cosa. Con arbol propio, el cache que se toca es
    ``_seleccion/labels/val.cache`` y el del dataset queda intacto.

    Se enlaza en duro en vez de copiar: mismo volumen, sin duplicar bytes.
    """
    datos_yaml = Path(datos_yaml)
    reales = [p for p in _imagenes_de_val(datos_yaml)
              if p.name.startswith(PREFIJO_REAL)]
    if not reales:
        return None

    img_dir = destino_dir / "images" / "val"
    lbl_dir = destino_dir / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img in reales:
        if not img.exists():
            continue
        _enlazar(img, img_dir / img.name)
        etq = _etiqueta_de(img)
        if etq.exists():
            _enlazar(etq, lbl_dir / etq.name)

    cfg = yaml.safe_load(datos_yaml.read_text(encoding="utf-8")) or {}
    nuevo = {
        "path": str(destino_dir.resolve()),
        # El train no se usa al validar, pero Ultralytics espera la clave.
        # Apunta al mismo subconjunto para no arrastrar el split grande.
        "train": "images/val",
        "val": "images/val",
        "names": cfg.get("names", {}),
    }
    salida = destino_dir / "dataset_solo_real.yaml"
    salida.write_text(yaml.safe_dump(nuevo, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
    return salida


def candidatos(run_dir: Path) -> List[Path]:
    """Checkpoints a evaluar: best, last y los periodicos si los hay."""
    pesos = Path(run_dir) / "weights"
    if not pesos.is_dir():
        return []
    vistos, salida = set(), []
    for p in [pesos / "best.pt", pesos / "last.pt"] + sorted(pesos.glob("epoch*.pt")):
        if p.exists() and p.name not in vistos and p.name != NOMBRE_MEJOR_REAL:
            vistos.add(p.name)
            salida.append(p)
    return salida


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def elegir_mejor_en_real(run_dir: Path, datos_yaml: Path, imgsz: int, batch: int,
                         device: str,
                         log: Optional[Callable[[str], None]] = None,
                         ) -> Tuple[Optional[Path], List[Dict]]:
    """Evalua los checkpoints sobre sedimento real y copia el mejor.

    Devuelve (ruta de best_real.pt, tabla de resultados). Si no se puede evaluar
    nada, devuelve (None, []) y el entrenamiento sigue con su best.pt de siempre.
    """
    def _log(m: str) -> None:
        if log:
            log(m)

    run_dir = Path(run_dir)
    pesos = candidatos(run_dir)
    if not pesos:
        _log("[INFO] No hay checkpoints que evaluar; se conserva best.pt.")
        return None, []

    yaml_real = crear_yaml_solo_real(Path(datos_yaml), run_dir / "_seleccion")
    if yaml_real is None:
        _log("[INFO] La validacion no tiene imagenes de sedimento real; "
             "se conserva best.pt.")
        return None, []

    from ultralytics import YOLO

    _log("")
    _log("=" * 60)
    _log("  ELIGIENDO CHECKPOINT SOBRE SEDIMENTO REAL")
    _log("=" * 60)
    _log(f"  Se evaluan {len(pesos)} checkpoint(s) contra solo las imagenes")
    _log("  de sedimento real de la validacion. best.pt viene elegido por el")
    _log("  mAP global, que en este dataset lo dominan las placas de laboratorio.")

    tabla: List[Dict] = []
    for p in pesos:
        try:
            res = YOLO(str(p)).val(data=str(yaml_real), imgsz=imgsz, batch=batch,
                                   device=device, plots=False, verbose=False)
            pr, rc = float(res.box.mp), float(res.box.mr)
            fila = {"checkpoint": p.name, "ruta": p,
                    "precision": pr, "recall": rc, "f1": _f1(pr, rc),
                    "map50": float(res.box.map50)}
        except Exception as exc:
            _log(f"  [WARN] no se pudo evaluar {p.name}: {exc}")
            continue
        tabla.append(fila)
        _log(f"    {fila['checkpoint']:<14} F1={fila['f1']:.4f}  "
             f"P={pr:.4f}  R={rc:.4f}  mAP50={fila['map50']:.4f}")

    if not tabla:
        _log("  No se pudo evaluar ningun checkpoint; se conserva best.pt.")
        return None, []

    # F1 como criterio, mAP50 como desempate: el conteo del paper depende de
    # acertar cuantas particulas hay, no de afinar la caja.
    tabla.sort(key=lambda d: (d["f1"], d["map50"]), reverse=True)
    mejor = tabla[0]

    destino = run_dir / "weights" / NOMBRE_MEJOR_REAL
    try:
        import shutil
        shutil.copy2(mejor["ruta"], destino)
    except OSError as exc:
        _log(f"  [WARN] no se pudo copiar el ganador: {exc}")
        return None, tabla

    _log("")
    _log(f"  Mejor en sedimento real: {mejor['checkpoint']} (F1={mejor['f1']:.4f})")
    origen_best = next((d for d in tabla if d["checkpoint"] == "best.pt"), None)
    if origen_best and origen_best["checkpoint"] != mejor["checkpoint"]:
        _log(f"  best.pt daba F1={origen_best['f1']:.4f}: la reeleccion cambia el peso.")
    elif origen_best:
        _log("  Coincide con best.pt: no habia nada mejor que elegir.")
    _log(f"  Guardado como {NOMBRE_MEJOR_REAL} (best.pt se conserva intacto).")
    _log("=" * 60)
    return destino, tabla
