"""Página 2 — Dataset. Carga data.yaml, valida estructura, vista previa, auto-split."""
from __future__ import annotations
from pathlib import Path
import random
import shutil

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QFrame, QSpinBox, QComboBox, QProgressDialog,
)

from ._base import TrainerPage
from ...core import theme as T
from ...core.paths import IMAGE_EXTS


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, d: dict):
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)


def _resolve_split_path(yaml_dir: Path, data: dict, split_value) -> Path | None:
    """Resuelve la ruta de un split aplicando la lógica de Ultralytics.

    Convención YOLO:
        path: ../datasets/coco128     <- raíz (opcional)
        train: images/train2017       <- relativo a 'path' si existe, si no a yaml_dir
        val:   images/val2017
    """
    if not split_value:
        return None
    p = Path(split_value)
    if p.is_absolute():
        return p
    root_value = data.get("path")
    if root_value:
        root = Path(root_value)
        if not root.is_absolute():
            root = (yaml_dir / root).resolve()
        candidate = (root / p).resolve()
        if candidate.exists():
            return candidate
        # Si no existe relativo a 'path', intentar relativo al yaml
    return (yaml_dir / p).resolve()


def _count_split(yaml_dir: Path, data: dict, split_value) -> int:
    p = _resolve_split_path(yaml_dir, data, split_value)
    if p is None:
        return 0
    if p.is_dir():
        n = 0
        for ext in IMAGE_EXTS:
            n += sum(1 for _ in p.rglob(f"*{ext}"))
        return n
    if p.is_file():
        # archivo de listado .txt con paths a imágenes
        try:
            return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        except Exception:
            return 0
    return 0


def _find_label_for(img_path: Path) -> Path | None:
    """Encuentra el .txt YOLO de una imagen, siguiendo la convención Ultralytics:
    reemplaza el segmento 'images' por 'labels' en la ruta."""
    # 1) Reemplazo de 'images' -> 'labels' en la ruta
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            new_parts = parts[:i] + ["labels"] + parts[i+1:]
            candidate = Path(*new_parts).with_suffix(".txt")
            if candidate.exists():
                return candidate
            break
    # 2) Mismo directorio
    same = img_path.with_suffix(".txt")
    if same.exists():
        return same
    # 3) labels/ hermana del padre
    sibling = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if sibling.exists():
        return sibling
    # 4) labels/ dentro del mismo directorio
    nested = img_path.parent / "labels" / (img_path.stem + ".txt")
    if nested.exists():
        return nested
    return None


class _PreviewTile(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(220, 150)
        self.setStyleSheet(
            f"background: {T.BG_SOFT}; border: 1px solid {T.RULE}; border-radius: 6px;"
        )
        self.setAlignment(Qt.AlignCenter)
        self.setText("(vacío)")


class DatasetPage(TrainerPage):
    PAGE_ICON = "📂"
    PAGE_TITLE = "Dataset"
    PAGE_DESCRIPTION = (
        "Carga tu data.yaml. Validamos la estructura, contamos imágenes y mostramos una "
        "vista previa con cajas. Si tu dataset no está dividido, usa Auto-split."
    )

    def __init__(self, state, parent=None):
        super().__init__(state, parent)

        # ── Origen ──
        c1, l1 = self.card("data.yaml", "📑")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.ed_yaml = QLineEdit()
        self.ed_yaml.setPlaceholderText("Ruta a data.yaml…")
        self.ed_yaml.editingFinished.connect(self._reload_yaml)
        row.addWidget(self.ed_yaml, 1)
        btn = QPushButton("…")
        btn.setFixedWidth(36)
        btn.clicked.connect(self._browse)
        row.addWidget(btn)
        l1.addLayout(row)

        self.lbl_yaml_status = QLabel("Sin data.yaml cargado.")
        self.lbl_yaml_status.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l1.addWidget(self.lbl_yaml_status)
        self.body.addWidget(c1)

        # ── Conteos ──
        c2, l2 = self.card("Conteos por split", "📊")
        grid = QGridLayout(); grid.setHorizontalSpacing(20); grid.setVerticalSpacing(8)
        self.lbl_train = QLabel("Train: —"); grid.addWidget(self.lbl_train, 0, 0)
        self.lbl_val   = QLabel("Val: —");   grid.addWidget(self.lbl_val,   0, 1)
        self.lbl_test  = QLabel("Test: —");  grid.addWidget(self.lbl_test,  0, 2)
        self.lbl_classes = QLabel("Clases: —"); grid.addWidget(self.lbl_classes, 1, 0, 1, 3)
        for w in (self.lbl_train, self.lbl_val, self.lbl_test, self.lbl_classes):
            w.setStyleSheet(f"color: {T.INK2}; font-size: 11pt; border: none;")
        l2.addLayout(grid)
        self.body.addWidget(c2)

        # ── Vista previa ──
        c3, l3 = self.card("Vista previa (6 imágenes random con cajas)", "👁")
        prev_grid = QGridLayout(); prev_grid.setSpacing(10)
        self.tiles: list[_PreviewTile] = []
        for i in range(6):
            t = _PreviewTile()
            self.tiles.append(t)
            prev_grid.addWidget(t, i // 3, i % 3)
        l3.addLayout(prev_grid)
        btn_refresh = QPushButton("🔄  Otras 6")
        btn_refresh.clicked.connect(self._refresh_preview)
        l3.addWidget(btn_refresh, 0, Qt.AlignLeft)
        self.body.addWidget(c3)

        # ── Auto-split ──
        c4, l4 = self.card("Auto-split (si tu dataset no está dividido)", "🪚")
        info = QLabel(
            "Toma una carpeta con images/ y labels/ y genera train/val/test automáticamente "
            "con un nuevo data.yaml. No mueve archivos: usa listas .txt."
        )
        info.setWordWrap(True); info.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l4.addWidget(info)

        srow = QHBoxLayout()
        srow.setSpacing(8)
        self.ed_split_src = QLineEdit()
        self.ed_split_src.setPlaceholderText("Carpeta raíz con images/ y labels/…")
        srow.addWidget(self.ed_split_src, 1)
        bbtn = QPushButton("…"); bbtn.setFixedWidth(36)
        bbtn.clicked.connect(self._browse_split_src); srow.addWidget(bbtn)
        l4.addLayout(srow)

        prow = QHBoxLayout(); prow.setSpacing(12)
        prow.addWidget(QLabel("Train %:"))
        self.sb_train = QSpinBox(); self.sb_train.setRange(50, 95); self.sb_train.setValue(80)
        prow.addWidget(self.sb_train)
        prow.addWidget(QLabel("Val %:"))
        self.sb_val = QSpinBox(); self.sb_val.setRange(5, 30); self.sb_val.setValue(15)
        prow.addWidget(self.sb_val)
        prow.addWidget(QLabel("Test %:"))
        self.sb_test = QSpinBox(); self.sb_test.setRange(0, 25); self.sb_test.setValue(5)
        prow.addWidget(self.sb_test)
        prow.addStretch(1)
        btn_split = QPushButton("🪚  Auto-split + generar data.yaml")
        btn_split.setStyleSheet(
            f"background: {T.ACCENT}; color: white; border: none; "
            f"border-radius: 6px; padding: 8px 16px; font-weight: 600;"
        )
        btn_split.setCursor(Qt.PointingHandCursor)
        btn_split.clicked.connect(self._auto_split)
        prow.addWidget(btn_split)
        l4.addLayout(prow)
        self.body.addWidget(c4)

        # ── Validación del dataset ──
        c5, l5 = self.card("Validación del dataset", "🔍")
        info_val = QLabel(
            "Verifica que el dataset sea correcto antes de entrenar. "
            "Se comprueba automáticamente al cargar data.yaml."
        )
        info_val.setWordWrap(True)
        info_val.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        l5.addWidget(info_val)

        # Labels de validación
        self._val_checks: list[QLabel] = []
        val_checks_names = [
            "data.yaml cargado y válido",
            "Split train con imágenes",
            "Split val con imágenes",
            "Labels encontradas (≥ 80 %)",
            "Clases definidas en YAML",
        ]
        val_grid = QGridLayout(); val_grid.setSpacing(6)
        for i, name in enumerate(val_checks_names):
            icon = QLabel("○")
            icon.setFixedWidth(20)
            icon.setStyleSheet(f"color: {T.INK3}; font-size: 11pt; border: none;")
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
            val_grid.addWidget(icon, i, 0)
            val_grid.addWidget(lbl, i, 1)
            self._val_checks.append((icon, lbl))
        l5.addLayout(val_grid)

        btn_val = QPushButton("🔍  Validar ahora")
        btn_val.clicked.connect(self._validate_dataset)
        l5.addWidget(btn_val, 0, Qt.AlignLeft)
        self.body.addWidget(c5)

    # ──────────────────────────────────────────
    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar data.yaml", "", "YAML (*.yaml *.yml)"
        )
        if f:
            self.ed_yaml.setText(f)
            self._reload_yaml()

    def _browse_split_src(self):
        d = QFileDialog.getExistingDirectory(self, "Carpeta raíz con images/ y labels/")
        if d:
            self.ed_split_src.setText(d)

    def _reload_yaml(self):
        t = self.ed_yaml.text().strip()
        if not t:
            return
        p = Path(t)
        if not p.exists():
            self.lbl_yaml_status.setText("✗ Archivo no encontrado.")
            self.lbl_yaml_status.setStyleSheet(f"color: {T.ERR}; font-size: 10pt; border: none;")
            return
        try:
            d = _load_yaml(p)
        except Exception as e:
            self.lbl_yaml_status.setText(f"✗ Error parseando YAML: {e}")
            self.lbl_yaml_status.setStyleSheet(f"color: {T.ERR}; font-size: 10pt; border: none;")
            return

        yaml_dir = p.parent
        train_p = d.get("train"); val_p = d.get("val"); test_p = d.get("test")
        n_train = _count_split(yaml_dir, d, train_p)
        n_val = _count_split(yaml_dir, d, val_p)
        n_test = _count_split(yaml_dir, d, test_p)

        names = d.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]

        self.state.dataset.yaml_path = p
        self.state.dataset.train_count = n_train
        self.state.dataset.val_count = n_val
        self.state.dataset.test_count = n_test
        self.state.dataset.class_names = list(names) if names else []
        self.state.dataset_changed.emit()

        self.lbl_yaml_status.setText(f"✓ {p}")
        self.lbl_yaml_status.setStyleSheet(f"color: {T.OK}; font-size: 10pt; border: none;")
        self.lbl_train.setText(f"Train: {n_train}")
        self.lbl_val.setText(f"Val: {n_val}")
        self.lbl_test.setText(f"Test: {n_test}")
        self.lbl_classes.setText(
            f"Clases ({len(self.state.dataset.class_names)}): "
            + ", ".join(self.state.dataset.class_names[:20])
        )
        self._refresh_preview()
        self._validate_dataset()

    def _refresh_preview(self):
        yaml_path = self.state.dataset.yaml_path
        if not yaml_path:
            return
        try:
            d = _load_yaml(yaml_path)
        except Exception:
            return
        yaml_dir = yaml_path.parent

        # raíz para resolver paths relativos dentro de listas .txt
        root_value = d.get("path")
        if root_value:
            root = Path(root_value)
            if not root.is_absolute():
                root = (yaml_dir / root).resolve()
        else:
            root = yaml_dir

        train_p = d.get("train")
        if not train_p:
            return
        train_abs = _resolve_split_path(yaml_dir, d, train_p)
        if train_abs is None or not train_abs.exists():
            # Mostrar mensaje en los tiles
            for tile in self.tiles:
                tile.setPixmap(QPixmap())
                tile.setText("(no se encontró\nla carpeta de train)")
            return

        imgs: list[Path] = []
        if train_abs.is_dir():
            for ext in IMAGE_EXTS:
                imgs.extend(list(train_abs.rglob(f"*{ext}"))[:200])
        elif train_abs.is_file():
            for line in train_abs.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pp = Path(line)
                if not pp.is_absolute():
                    # Ultralytics resuelve los listados respecto a 'path' o al yaml
                    pp = (root / pp).resolve()
                    if not pp.exists():
                        pp = (yaml_dir / line).resolve()
                if pp.exists():
                    imgs.append(pp)
        if not imgs:
            for tile in self.tiles:
                tile.setPixmap(QPixmap())
                tile.setText("(sin imágenes\nen train)")
            return

        random.shuffle(imgs)
        sample = imgs[:6]
        for tile, img_path in zip(self.tiles, sample):
            self._draw_preview(tile, img_path)
        for tile in self.tiles[len(sample):]:
            tile.setPixmap(QPixmap()); tile.setText("(vacío)")

    def _draw_preview(self, tile: _PreviewTile, img_path: Path):
        pm = QPixmap(str(img_path))
        if pm.isNull():
            tile.setText("✗")
            return
        # Buscar .txt YOLO con la convención Ultralytics
        boxes = []
        label_path = _find_label_for(img_path)
        if label_path is not None:
            try:
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(float(parts[0]))
                        cx, cy, w, h = [float(x) for x in parts[1:5]]
                        boxes.append((cid, cx, cy, w, h))
            except Exception:
                pass
        # dibujar
        img = QImage(pm)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        names = self.state.dataset.class_names
        for cid, cx, cy, w, h in boxes:
            name = names[cid] if cid < len(names) else str(cid)
            color = QColor(T.CLASS_COLOR_HEX.get(name, "#33aaff"))
            painter.setPen(QPen(color, max(2, img.width() // 300)))
            x = (cx - w / 2) * img.width(); y = (cy - h / 2) * img.height()
            painter.drawRect(int(x), int(y), int(w * img.width()), int(h * img.height()))
        painter.end()
        scaled = QPixmap.fromImage(img).scaled(tile.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        tile.setPixmap(scaled)

    # ── Validación ────────────────────────────────────────────
    def _set_check(self, idx: int, ok: bool | None, msg: str = ""):
        icon_lbl, text_lbl = self._val_checks[idx]
        if ok is True:
            icon_lbl.setText("✓")
            icon_lbl.setStyleSheet(f"color: {T.OK}; font-size: 11pt; border: none;")
            text_lbl.setStyleSheet(f"color: {T.OK}; font-size: 10pt; border: none;")
        elif ok is False:
            icon_lbl.setText("✗")
            icon_lbl.setStyleSheet(f"color: {T.ERR}; font-size: 11pt; border: none;")
            text_lbl.setStyleSheet(f"color: {T.ERR}; font-size: 10pt; border: none;")
        else:
            icon_lbl.setText("○")
            icon_lbl.setStyleSheet(f"color: {T.INK3}; font-size: 11pt; border: none;")
            text_lbl.setStyleSheet(f"color: {T.INK3}; font-size: 10pt; border: none;")
        if msg:
            text_lbl.setToolTip(msg)

    def _validate_dataset(self):
        """Valida el dataset y actualiza los indicadores de validación."""
        # Reset
        for i in range(len(self._val_checks)):
            self._set_check(i, None)

        yaml_path = self.state.dataset.yaml_path
        if not yaml_path or not yaml_path.exists():
            self._set_check(0, False, "No hay data.yaml cargado.")
            return
        try:
            d = _load_yaml(yaml_path)
        except Exception as e:
            self._set_check(0, False, str(e))
            return
        self._set_check(0, True)

        yaml_dir = yaml_path.parent
        n_train = self.state.dataset.train_count
        n_val   = self.state.dataset.val_count
        self._set_check(1, n_train > 0,
                        f"{n_train} imagen(es) en train." if n_train else "Sin imágenes en train.")
        self._set_check(2, n_val > 0,
                        f"{n_val} imagen(es) en val." if n_val else "Sin imágenes en val.")

        # Verificar labels para las primeras 50 imágenes de train
        train_p = d.get("train")
        train_abs = _resolve_split_path(yaml_dir, d, train_p)
        if train_abs and train_abs.exists():
            imgs: list[Path] = []
            if train_abs.is_dir():
                for ext in IMAGE_EXTS:
                    imgs.extend(list(train_abs.rglob(f"*{ext}"))[:50])
            if imgs:
                with_label = sum(1 for img in imgs if _find_label_for(img))
                pct = with_label / len(imgs) * 100
                ok_labels = pct >= 80
                self._set_check(3, ok_labels,
                                f"{pct:.0f} % de imágenes tienen label ({with_label}/{len(imgs)} revisadas).")
            else:
                self._set_check(3, None, "No se encontraron imágenes para revisar.")
        else:
            self._set_check(3, None, "No se pudo acceder al split train.")

        names = d.get("names", [])
        if isinstance(names, dict):
            names = list(names.values())
        has_names = bool(names)
        self._set_check(4, has_names,
                        f"Clases: {', '.join(str(n) for n in names[:10])}" if has_names
                        else "No hay clave 'names' en el YAML.")

    # ── Auto-split ────────────────────────────────────────────
    def _auto_split(self):
        src = self.ed_split_src.text().strip()
        if not src:
            QMessageBox.warning(self, "Auto-split", "Selecciona la carpeta raíz primero.")
            return
        root = Path(src)
        imgs_dir = root / "images"
        if not imgs_dir.exists():
            QMessageBox.warning(self, "Auto-split",
                                f"No existe {imgs_dir}. Necesitas images/ y labels/ dentro de la carpeta.")
            return
        # recolectar imágenes
        all_imgs: list[Path] = []
        for ext in IMAGE_EXTS:
            all_imgs.extend(imgs_dir.rglob(f"*{ext}"))
        if not all_imgs:
            QMessageBox.warning(self, "Auto-split", "No se encontraron imágenes en images/.")
            return

        tp, vp, sp = self.sb_train.value(), self.sb_val.value(), self.sb_test.value()
        if tp + vp + sp > 100:
            QMessageBox.warning(self, "Auto-split", "Los porcentajes suman > 100.")
            return

        random.seed(42)
        random.shuffle(all_imgs)
        N = len(all_imgs)
        n_tr = int(N * tp / 100)
        n_vl = int(N * vp / 100)
        train_list = all_imgs[:n_tr]
        val_list   = all_imgs[n_tr:n_tr + n_vl]
        test_list  = all_imgs[n_tr + n_vl:] if sp > 0 else []

        # Guardar listas
        for fname, lst in [("train.txt", train_list), ("val.txt", val_list), ("test.txt", test_list)]:
            if not lst: continue
            (root / fname).write_text(
                "\n".join(str(p.relative_to(root)) for p in lst), encoding="utf-8"
            )

        # Leer classes.txt si existe
        cls_path = root / "classes.txt"
        names: list[str] = []
        if cls_path.exists():
            names = [l.strip() for l in cls_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        d = {
            "path": str(root),
            "train": "train.txt",
            "val": "val.txt",
        }
        if test_list: d["test"] = "test.txt"
        if names:     d["names"] = names

        yaml_out = root / "data.yaml"
        _save_yaml(yaml_out, d)
        QMessageBox.information(
            self, "Auto-split",
            f"Listo:\n• Train: {len(train_list)}\n• Val: {len(val_list)}\n"
            f"• Test: {len(test_list)}\n\nGenerado: {yaml_out}"
        )
        self.ed_yaml.setText(str(yaml_out))
        self._reload_yaml()
