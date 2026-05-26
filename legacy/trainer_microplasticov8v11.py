import os
import sys
import shutil
import yaml
import datetime
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import multiprocessing as mp
import contextlib

import torch
from ultralytics import YOLO


# =========================
#  Autoría
# =========================
AUTHOR_NAME = "Cristofher Ferrada"
AUTHOR_YEAR = "2026"


# =========================
#  Estimación VRAM (aprox)
# =========================
BASE_MEM_MODELS = {
    # YOLOv8
    "yolov8n.pt": 3.0, "yolov8s.pt": 4.5, "yolov8m.pt": 7.0, "yolov8l.pt": 9.0, "yolov8x.pt": 12.0,
    # YOLOv11 (referencial; calibra)
    "yolo11n.pt": 3.2, "yolo11s.pt": 4.8, "yolo11m.pt": 7.5, "yolo11l.pt": 9.8, "yolo11x.pt": 12.8,
    # Alternativo común
    "yolov11n.pt": 3.2, "yolov11s.pt": 4.8, "yolov11m.pt": 7.5, "yolov11l.pt": 9.8, "yolov11x.pt": 12.8,
}

FAMILIES = {
    # Si tus pesos YOLOv11 se llaman "yolov11x.pt" en vez de "yolo11x.pt",
    # cambia prefix a "yolov11".
    "YOLOv8":  {"prefix": "yolov8", "sizes": ["n", "s", "m", "l", "x"]},
    "YOLOv11": {"prefix": "yolo11", "sizes": ["n", "s", "m", "l", "x"]},
}


# =========================
#  Utils
# =========================
def get_gpu_info():
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / (1024 ** 3)
    name = props.name
    return {"name": name, "total_gb": total_gb, "device_index": device}


def estimate_memory_gb(model_name, img_size, batch_size):
    base = BASE_MEM_MODELS.get(model_name, 4.0)
    scale_batch = batch_size / 16
    scale_img = (img_size / 640) ** 2
    return base * scale_batch * scale_img


def get_augmentation_params(level: str):
    level = (level or "").lower()
    if "sin" in level:
        return dict(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0)
    if "suave" in level:
        return dict(degrees=10.0, translate=0.02, scale=0.10, shear=0.0, flipud=0.2, fliplr=0.5, mosaic=0.0)
    if "fuerte" in level:
        return dict(degrees=25.0, translate=0.10, scale=0.30, shear=0.0, flipud=0.5, fliplr=0.5, mosaic=1.0)
    return dict(degrees=15.0, translate=0.05, scale=0.20, shear=0.0, flipud=0.3, fliplr=0.5, mosaic=0.5)


def resolve_dataset_root(yaml_path: str) -> str:
    y = os.path.abspath(yaml_path)
    cfg = yaml.safe_load(open(y, "r", encoding="utf-8")) or {}
    base_dir = os.path.dirname(y)
    rel = cfg.get("path", ".")
    root = os.path.abspath(os.path.join(base_dir, rel))
    return root


def validate_dataset_yaml(yaml_path: str):
    try:
        y = os.path.abspath(yaml_path)
        cfg = yaml.safe_load(open(y, "r", encoding="utf-8")) or {}
        root = resolve_dataset_root(yaml_path)

        checks = {
            "images/train": os.path.isdir(os.path.join(root, "images", "train")),
            "images/val":   os.path.isdir(os.path.join(root, "images", "val")),
            "labels/train": os.path.isdir(os.path.join(root, "labels", "train")),
            "labels/val":   os.path.isdir(os.path.join(root, "labels", "val")),
        }
        missing = [k for k, ok in checks.items() if not ok]

        names = cfg.get("names", None)
        names_note = "OK: 'names' presente." if names is not None else "⚠️ Nota: 'names:' no está en el YAML (recomendado)."

        if missing:
            return False, (
                f"dataset.yaml: {y}\n"
                f"Dataset root resuelto: {root}\n"
                f"Faltan carpetas: {', '.join(missing)}\n"
                f"{names_note}\n\n"
                f"Tip: deja el YAML dentro del dataset y usa 'path: .' para que sea robusto."
            )

        return True, (
            f"OK ✅ dataset.yaml válido\n"
            f"dataset.yaml: {y}\n"
            f"Dataset root resuelto: {root}\n"
            f"{names_note}"
        )
    except Exception as e:
        return False, f"Error leyendo/validando YAML: {e}"


def is_probably_weight_name(s: str) -> bool:
    s = s.strip().lower()
    return ("/" not in s and "\\" not in s and s.endswith(".pt"))


def ensure_weights_available(weights: str) -> str:
    """
    Si es ruta existente -> OK.
    Si es nombre tipo yolo11x.pt y no existe -> intenta que Ultralytics lo descargue al instanciar YOLO(nombre).
    """
    w = weights.strip()

    if os.path.isfile(w):
        return w

    if is_probably_weight_name(w):
        _ = YOLO(w)  # si Ultralytics soporta ese nombre, lo baja/resuelve
        return w

    raise FileNotFoundError(f"Ruta de weights no existe: {w}")


# =========================
#  Proceso de entrenamiento (separado)
# =========================
def train_worker(proc_args: dict):
    """
    Corre en proceso separado.
    Escribe log en proc_args['log_path'].
    """
    log_path = proc_args["log_path"]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        with open(log_path, "a", encoding="utf-8") as f, \
             contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            print("========== TRAIN WORKER START ==========")
            print(f"Autor: {AUTHOR_NAME}")
            print(f"Año: {AUTHOR_YEAR}")
            print("Hora:", datetime.datetime.now().isoformat(timespec="seconds"))
            print("Args:", proc_args)
            print("")

            data_path = proc_args["data_path"]
            ok, msg = validate_dataset_yaml(data_path)
            print(msg)
            if not ok:
                raise RuntimeError("Dataset inválido. Abortando entrenamiento.")

            # Pesos (auto-download si aplica)
            weights = ensure_weights_available(proc_args["weights"])

            model = YOLO(weights)
            aug_params = get_augmentation_params(proc_args["aug_level"])

            model.train(
                data=data_path,
                imgsz=proc_args["imgsz"],
                epochs=proc_args["epochs"],
                batch=proc_args["batch"],
                lr0=proc_args["lr0"],
                project=proc_args["project_dir"],
                name=proc_args["run_name"],
                device=proc_args["device"],
                workers=proc_args["workers"],
                patience=proc_args["patience"],
                cos_lr=proc_args["cos_lr"],
                amp=proc_args["amp"],
                **aug_params,
            )

            print("\n===== ENTRENAMIENTO FINALIZADO =====")
            run_dir = os.path.join(proc_args["project_dir"], proc_args["run_name"])
            best_path = os.path.join(run_dir, "weights", "best.pt")
            print(f"Run dir: {run_dir}")
            print(f"best.pt: {best_path}")
            print(f"Autor: {AUTHOR_NAME} ({AUTHOR_YEAR})")

    except Exception:
        tb = traceback.format_exc()
        log("\n[ERROR EN TRAIN WORKER]")
        log(tb)


# =========================
#  GUI
# =========================
class YOLOTrainerProGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"YOLO Trainer Pro — Microplásticos (PET/PP/LDPE) — {AUTHOR_NAME} {AUTHOR_YEAR}")
        self.root.geometry("1140x780")
        self.root.minsize(1020, 700)

        self.proc: mp.Process | None = None
        self.is_training = False
        self.is_paused = False
        self.train_start_time: datetime.datetime | None = None

        self.log_path: str | None = None
        self._log_fp = None
        self._log_pos = 0

        self._setup_style()
        self._build_ui()
        self._update_gpu_label()
        self._refresh_model_from_family_size()
        self._set_status("Listo.")

        self.root.after(300, self._poll_log_and_process)

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TEntry", font=("Segoe UI", 10))
        style.configure("TCombobox", font=("Segoe UI", 10))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(12, 10))
        top.pack(fill=tk.X)

        ttk.Label(top, text="YOLO Trainer Pro", style="Header.TLabel").pack(side=tk.LEFT)
        self.gpu_label = ttk.Label(top, text="GPU: -")
        self.gpu_label.pack(side=tk.RIGHT)

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        tab_train = ttk.Frame(nb, padding=10)
        tab_cfg = ttk.Frame(nb, padding=10)
        nb.add(tab_train, text="Entrenamiento")
        nb.add(tab_cfg, text="Notas")

        grid = ttk.Frame(tab_train)
        grid.pack(fill=tk.BOTH, expand=True)
        grid.columnconfigure(0, weight=0)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)

        left = ttk.Labelframe(grid, text="Parámetros", padding=10)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        right = ttk.Labelframe(grid, text="Log", padding=10)
        right.grid(row=0, column=1, sticky="nsew")

        r = 0

        ttk.Label(left, text="Familia:").grid(row=r, column=0, sticky="w", pady=3)
        self.family_var = tk.StringVar(value="YOLOv11")
        self.family_cb = ttk.Combobox(left, textvariable=self.family_var, values=list(FAMILIES.keys()),
                                      state="readonly", width=12)
        self.family_cb.grid(row=r, column=1, sticky="w", pady=3)
        self.family_cb.bind("<<ComboboxSelected>>", lambda e: self._on_family_change())

        r += 1
        ttk.Label(left, text="Tamaño:").grid(row=r, column=0, sticky="w", pady=3)
        self.size_var = tk.StringVar(value="x")
        self.size_cb = ttk.Combobox(left, textvariable=self.size_var,
                                    values=FAMILIES[self.family_var.get()]["sizes"],
                                    state="readonly", width=12)
        self.size_cb.grid(row=r, column=1, sticky="w", pady=3)
        self.size_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_model_from_family_size())

        r += 1
        ttk.Label(left, text="Weights:").grid(row=r, column=0, sticky="w", pady=3)
        self.weights_var = tk.StringVar(value="")
        self.weights_entry = ttk.Entry(left, textvariable=self.weights_var, width=34)
        self.weights_entry.grid(row=r, column=1, sticky="w", pady=3)
        ttk.Button(left, text="Elegir...", command=self._browse_weights).grid(row=r, column=2, padx=6)

        r += 1
        self.auto_weights_var = tk.BooleanVar(value=True)
        self.auto_weights_chk = ttk.Checkbutton(
            left, text="Auto (familia+tamaño)", variable=self.auto_weights_var,
            command=self._refresh_model_from_family_size
        )
        self.auto_weights_chk.grid(row=r, column=1, sticky="w", pady=(0, 8))

        r += 1
        ttk.Label(left, text="imgsz:").grid(row=r, column=0, sticky="w", pady=3)
        self.imgsz_var = tk.IntVar(value=1280)
        ttk.Entry(left, textvariable=self.imgsz_var, width=12).grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        ttk.Label(left, text="epochs:").grid(row=r, column=0, sticky="w", pady=3)
        self.epochs_var = tk.IntVar(value=150)
        ttk.Entry(left, textvariable=self.epochs_var, width=12).grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        ttk.Label(left, text="batch:").grid(row=r, column=0, sticky="w", pady=3)
        self.batch_var = tk.IntVar(value=8)
        ttk.Entry(left, textvariable=self.batch_var, width=12).grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        ttk.Label(left, text="lr0:").grid(row=r, column=0, sticky="w", pady=3)
        self.lr0_var = tk.DoubleVar(value=0.01)
        ttk.Entry(left, textvariable=self.lr0_var, width=12).grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        ttk.Label(left, text="Aumento:").grid(row=r, column=0, sticky="w", pady=3)
        self.aug_var = tk.StringVar(value="Medio (recomendado)")
        ttk.Combobox(
            left, textvariable=self.aug_var,
            values=["Sin aumento extra", "Suave", "Medio (recomendado)", "Fuerte"],
            state="readonly", width=20
        ).grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        ttk.Label(left, text="dataset.yaml:").grid(row=r, column=0, sticky="w", pady=3)
        self.data_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.data_var, width=34).grid(row=r, column=1, sticky="w", pady=3)
        ttk.Button(left, text="Buscar...", command=self._browse_data_yaml).grid(row=r, column=2, padx=6)

        r += 1
        ttk.Label(left, text="Carpeta runs:").grid(row=r, column=0, sticky="w", pady=3)
        self.project_var = tk.StringVar(value="runs_microplastico")
        ttk.Entry(left, textvariable=self.project_var, width=34).grid(row=r, column=1, sticky="w", pady=3)
        ttk.Button(left, text="Elegir...", command=self._browse_project_dir).grid(row=r, column=2, padx=6)

        r += 1
        ttk.Label(left, text="Nombre exp:").grid(row=r, column=0, sticky="w", pady=3)
        self.name_var = tk.StringVar(value="")
        ttk.Entry(left, textvariable=self.name_var, width=34).grid(row=r, column=1, sticky="w", pady=3)
        ttk.Button(left, text="Sugerir", command=self._suggest_name).grid(row=r, column=2, padx=6)

        r += 1
        ttk.Label(left, text="Device:").grid(row=r, column=0, sticky="w", pady=3)
        self.device_var = tk.StringVar(value="auto")
        ttk.Combobox(left, textvariable=self.device_var, values=["auto", "0", "cpu"], state="readonly", width=12)\
            .grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        ttk.Label(left, text="workers:").grid(row=r, column=0, sticky="w", pady=3)
        self.workers_var = tk.IntVar(value=4)
        ttk.Entry(left, textvariable=self.workers_var, width=12).grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        ttk.Label(left, text="patience:").grid(row=r, column=0, sticky="w", pady=3)
        self.patience_var = tk.IntVar(value=50)
        ttk.Entry(left, textvariable=self.patience_var, width=12).grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        self.coslr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="cos_lr", variable=self.coslr_var).grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        self.amp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="AMP (mixed precision)", variable=self.amp_var).grid(row=r, column=1, sticky="w", pady=3)

        r += 1
        self.mem_label = ttk.Label(left, text="Memoria estimada: -")
        self.mem_label.grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 2))

        r += 1
        ttk.Button(left, text="Calcular memoria", command=self._calculate_memory).grid(
            row=r, column=0, columnspan=3, sticky="we", pady=5
        )

        r += 1
        self.btn_train = ttk.Button(left, text="▶ Iniciar entrenamiento", command=self._start_training)
        self.btn_train.grid(row=r, column=0, columnspan=3, sticky="we", pady=(8, 4))

        r += 1
        btn_row = ttk.Frame(left)
        btn_row.grid(row=r, column=0, columnspan=3, sticky="we")
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        self.btn_pause = ttk.Button(btn_row, text="⏸ Pausa (log)", command=self._toggle_pause)
        self.btn_pause.grid(row=0, column=0, sticky="we", padx=(0, 4))

        self.btn_stop = ttk.Button(btn_row, text="■ Detener (REAL)", command=self._stop_training_real)
        self.btn_stop.grid(row=0, column=1, sticky="we", padx=(4, 0))

        r += 1
        self.btn_open_run = ttk.Button(left, text="📁 Abrir carpeta runs", command=self._open_runs_folder)
        self.btn_open_run.grid(row=r, column=0, columnspan=3, sticky="we", pady=(8, 0))

        toolbar = ttk.Frame(right)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        self.progress = ttk.Progressbar(toolbar, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(toolbar, text="Copiar", command=self._copy_log).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(toolbar, text="Limpiar", command=lambda: self.log_text.delete("1.0", tk.END)).pack(side=tk.RIGHT)

        self.log_text = tk.Text(right, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        ttk.Label(tab_cfg, text="Notas rápidas", style="Header.TLabel").pack(anchor="w")
        notes = (
            f"• Autor: {AUTHOR_NAME} ({AUTHOR_YEAR}).\n"
            "• FIX Windows: el entrenamiento corre en proceso NO-daemon (para permitir DataLoader workers).\n"
            "• Auto weights: si no existe el .pt, el worker intenta descargarlo vía Ultralytics.\n"
            "• Pausa (log): pausa el streaming del log (no pausa el entrenamiento real).\n"
            "• Detener (REAL): termina el proceso de entrenamiento.\n"
        )
        ttk.Label(tab_cfg, text=notes, justify="left").pack(anchor="w", pady=(8, 0))

        # Status bar
        self.status = ttk.Label(self.root, text="", anchor="w")
        self.status.pack(fill=tk.X, padx=12, pady=(0, 0))

        # Footer autoría
        footer = ttk.Label(
            self.root,
            text=f"{AUTHOR_NAME} © {AUTHOR_YEAR} — YOLO Trainer Pro (Microplásticos)",
            anchor="center",
            font=("Segoe UI", 9)
        )
        footer.pack(side=tk.BOTTOM, pady=(0, 6))

    def _set_status(self, text: str):
        self.status.config(text=text)

    def _update_gpu_label(self):
        info = get_gpu_info()
        if info is None:
            self.gpu_label.config(text="GPU: CUDA no detectada (CPU)")
        else:
            self.gpu_label.config(text=f"GPU: {info['name']} | VRAM ~ {info['total_gb']:.1f} GB")

    def _on_family_change(self):
        fam = self.family_var.get()
        self.size_cb["values"] = FAMILIES[fam]["sizes"]
        if self.size_var.get() not in FAMILIES[fam]["sizes"]:
            self.size_var.set(FAMILIES[fam]["sizes"][0])
        self._refresh_model_from_family_size()

    def _refresh_model_from_family_size(self):
        if not hasattr(self, "auto_weights_var"):
            return
        if not self.auto_weights_var.get():
            return
        fam = self.family_var.get()
        size = self.size_var.get()
        prefix = FAMILIES[fam]["prefix"]
        self.weights_var.set(f"{prefix}{size}.pt")

    def _browse_weights(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar weights (.pt)",
            filetypes=[("PyTorch weights", "*.pt"), ("Todos", "*.*")]
        )
        if filename:
            self.auto_weights_var.set(False)
            self.weights_var.set(filename)

    def _browse_data_yaml(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar dataset.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("Todos", "*.*")]
        )
        if filename:
            self.data_var.set(filename)

    def _browse_project_dir(self):
        dirname = filedialog.askdirectory(title="Seleccionar carpeta de proyecto (runs)")
        if dirname:
            self.project_var.set(dirname)

    def _suggest_name(self):
        fam = self.family_var.get().replace("YOLO", "yolo").lower()
        size = self.size_var.get()
        img = self.imgsz_var.get()
        batch = self.batch_var.get()
        epochs = self.epochs_var.get()
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.name_var.set(f"MP_{fam}{size}_img{img}_b{batch}_e{epochs}_{now}")

    def _calculate_memory(self):
        try:
            model_name = self.weights_var.get().strip()
            img = int(self.imgsz_var.get())
            batch = int(self.batch_var.get())
        except ValueError:
            messagebox.showerror("Error", "Revisa imgsz y batch.")
            return

        key = os.path.basename(model_name)
        info = get_gpu_info()
        est = estimate_memory_gb(key, img, batch)

        if info is None:
            msg = f"Memoria estimada: ~{est:.1f} GB\nNo se detectó GPU."
        else:
            total = info["total_gb"]
            ratio = est / total
            if ratio < 0.6:
                estado = "✅ OK"
            elif ratio < 0.9:
                estado = "⚠️ Ajustado"
            else:
                estado = "❌ Probable OOM"
            msg = (
                f"Modelo: {key}\nimgsz: {img}\nbatch: {batch}\n"
                f"Memoria estimada: ~{est:.1f} GB\nVRAM: ~{total:.1f} GB\n{estado}"
            )

        self.mem_label.config(text="Memoria estimada: " + msg.replace("\n", " | "))
        messagebox.showinfo("Estimación de memoria", msg)

    def _copy_log(self):
        text = self.log_text.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Log copiado al portapapeles.")

    def _open_runs_folder(self):
        folder = self.project_var.get().strip() or "runs_microplastico"
        folder = os.path.abspath(folder)
        os.makedirs(folder, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore
            elif sys.platform == "darwin":
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _start_training(self):
        if self.is_training:
            messagebox.showwarning("En curso", "Ya hay un entrenamiento en ejecución.")
            return

        data_path = self.data_var.get().strip()
        if not os.path.isfile(data_path):
            messagebox.showerror("Error", "Selecciona un dataset.yaml válido.")
            return

        ok, msg = validate_dataset_yaml(data_path)
        if not ok:
            messagebox.showerror("Dataset inválido", msg)
            return

        weights = self.weights_var.get().strip()
        if not weights:
            messagebox.showerror("Error", "Indica weights (Auto o archivo).")
            return

        project_dir = self.project_var.get().strip()
        if not project_dir:
            messagebox.showerror("Error", "Indica carpeta runs.")
            return
        project_dir = os.path.abspath(project_dir)
        os.makedirs(project_dir, exist_ok=True)

        run_name = self.name_var.get().strip()
        if not run_name:
            self._suggest_name()
            run_name = self.name_var.get().strip()

        try:
            imgsz = int(self.imgsz_var.get())
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            lr0 = float(self.lr0_var.get())
            workers = int(self.workers_var.get())
            patience = int(self.patience_var.get())
        except ValueError:
            messagebox.showerror("Error", "Revisa imgsz/epochs/batch/lr0/workers/patience.")
            return

        device_choice = self.device_var.get()
        if device_choice == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"
        elif device_choice == "0":
            device = 0
        else:
            device = "cpu"

        cos_lr = bool(self.coslr_var.get())
        amp = bool(self.amp_var.get())
        aug_level = self.aug_var.get()

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(project_dir, run_name, f"gui_train_{stamp}.log")
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

        self._close_log_fp()
        self._log_pos = 0
        self.log_text.insert(tk.END, "\n========== INICIANDO ENTRENAMIENTO ==========\n")
        self.log_text.insert(tk.END, f"Autor: {AUTHOR_NAME} ({AUTHOR_YEAR})\n")
        self.log_text.insert(tk.END, f"Log: {self.log_path}\n")
        self.log_text.see(tk.END)

        proc_args = dict(
            weights=weights,
            data_path=os.path.abspath(data_path),
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            lr0=lr0,
            project_dir=project_dir,
            run_name=run_name,
            aug_level=aug_level,
            device=device,
            workers=workers,
            patience=patience,
            cos_lr=cos_lr,
            amp=amp,
            log_path=self.log_path,
        )

        # IMPORTANTE: NO daemon=True (para permitir workers del DataLoader)
        self.proc = mp.Process(target=train_worker, args=(proc_args,))
        self.proc.start()

        self.is_training = True
        self.is_paused = False
        self.train_start_time = datetime.datetime.now()

        self.progress.start(10)
        self.btn_pause.configure(text="⏸ Pausa (log)")
        self._set_status("Entrenando...")

    def _toggle_pause(self):
        if not self.is_training:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.progress.stop()
            self.btn_pause.configure(text="▶ Reanudar (log)")
            self._set_status("Pausado (solo log).")
        else:
            self.progress.start(10)
            self.btn_pause.configure(text="⏸ Pausa (log)")
            self._set_status("Entrenando...")

    def _stop_training_real(self):
        if not self.is_training or self.proc is None:
            return

        resp = messagebox.askyesno("Detener", "Esto terminará el entrenamiento.\n¿Seguro?")
        if not resp:
            return

        try:
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=3)
        except Exception as e:
            messagebox.showerror("Error al detener", str(e))

        self._finish_training_ui(stopped=True)

    def _finish_training_ui(self, stopped: bool):
        self.is_training = False
        self.is_paused = False
        self.progress.stop()

        if stopped:
            self._set_status("Entrenamiento detenido.")
        else:
            elapsed = datetime.datetime.now() - (self.train_start_time or datetime.datetime.now())
            self._set_status(f"Finalizado. Tiempo: {str(elapsed).split('.')[0]}")

        if self.log_path and not stopped:
            run_dir = str(Path(self.log_path).parent)
            best_path = os.path.join(run_dir, "weights", "best.pt")
            if os.path.isfile(best_path):
                self._ask_save_trained_model(best_path)

    def _ask_save_trained_model(self, best_path: str):
        resp = messagebox.askyesno("Guardar modelo", f"¿Guardar copia de best.pt?\n\n{best_path}")
        if not resp:
            return

        dest = filedialog.asksaveasfilename(
            title="Guardar modelo entrenado",
            defaultextension=".pt",
            initialfile=os.path.basename(best_path),
            filetypes=[("PyTorch weights", "*.pt"), ("Todos", "*.*")]
        )
        if not dest:
            return

        try:
            shutil.copy2(best_path, dest)
            messagebox.showinfo("Guardado", f"Modelo guardado en:\n{dest}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _close_log_fp(self):
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None

    def _poll_log_and_process(self):
        if self.log_path and not self.is_paused:
            try:
                if self._log_fp is None:
                    self._log_fp = open(self.log_path, "r", encoding="utf-8", errors="replace")
                    self._log_fp.seek(self._log_pos)

                chunk = self._log_fp.read()
                if chunk:
                    self._log_pos = self._log_fp.tell()
                    self.log_text.insert(tk.END, chunk)
                    self.log_text.see(tk.END)
            except FileNotFoundError:
                pass
            except Exception:
                pass

        if self.is_training and self.proc is not None:
            if not self.proc.is_alive():
                try:
                    self.proc.join(timeout=0.2)
                except Exception:
                    pass
                self.proc = None
                self._finish_training_ui(stopped=False)

        if self.is_training and self.train_start_time:
            elapsed = datetime.datetime.now() - self.train_start_time
            if not self.is_paused:
                self._set_status(f"Entrenando... Tiempo: {str(elapsed).split('.')[0]}")
            else:
                self._set_status(f"Pausado (log). Tiempo: {str(elapsed).split('.')[0]}")

        self.root.after(300, self._poll_log_and_process)


def main():
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    root = tk.Tk()
    app = YOLOTrainerProGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()