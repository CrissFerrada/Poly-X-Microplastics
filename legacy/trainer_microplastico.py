import os
import threading
import datetime
import sys
import io
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
from ultralytics import YOLO
import yaml


# ====== Utilidades de GPU y memoria estimada ======

BASE_MEM_MODELS = {
    "yolov8n.pt": 3.0,
    "yolov8s.pt": 4.5,
    "yolov8m.pt": 7.0,
    "yolov8l.pt": 9.0,
    "yolov8x.pt": 12.0,
}


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


# ====== Augmentación según nivel (Microplásticos) ======

def get_augmentation_params(level: str):
    """
    Microplásticos:
    - Rotación/flip suele ser válido (no hay orientación "correcta").
    - Mosaic puede ayudar si hay objetos pequeños y densos, pero puede distorsionar fondos.
    """
    level = (level or "").lower()

    if "sin" in level:
        return dict(
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
        )

    if "suave" in level:
        return dict(
            degrees=10.0,
            translate=0.02,
            scale=0.10,
            shear=0.0,
            flipud=0.2,
            fliplr=0.5,
            mosaic=0.0,
        )

    if "fuerte" in level:
        return dict(
            degrees=25.0,
            translate=0.10,
            scale=0.30,
            shear=0.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
        )

    # Medio (por defecto)
    return dict(
        degrees=15.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.5,
    )


# ====== Validación y resolución robusta del dataset.yaml ======

def resolve_dataset_root(yaml_path: str) -> str:
    """
    Resuelve la carpeta raíz real del dataset según 'path:' dentro del YAML.
    Si el YAML tiene path: . y está dentro de data_microplastico, entonces root=data_microplastico.
    """
    y = os.path.abspath(yaml_path)
    cfg = yaml.safe_load(open(y, "r", encoding="utf-8")) or {}
    base_dir = os.path.dirname(y)          # carpeta donde vive el dataset.yaml
    rel = cfg.get("path", ".")             # valor de 'path:' (puede ser '.' o 'data_microplastico', etc.)
    root = os.path.abspath(os.path.join(base_dir, rel))
    return root


def validate_dataset_yaml(yaml_path: str) -> tuple[bool, str]:
    """
    Verifica que existan las carpetas mínimas:
    images/train, images/val, labels/train, labels/val
    """
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

        # También chequea que names exista (no es obligatorio para correr, pero sí recomendable)
        names = cfg.get("names", None)
        if names is None:
            names_note = "⚠️ Nota: 'names:' no está en el YAML (recomendado agregarlo)."
        else:
            names_note = "OK: 'names' presente."

        if missing:
            return False, (
                f"dataset.yaml: {y}\n"
                f"Dataset root resuelto: {root}\n"
                f"Faltan carpetas: {', '.join(missing)}\n"
                f"{names_note}\n\n"
                f"Tip: lo más robusto es tener el YAML dentro de la carpeta del dataset y usar 'path: .'."
            )

        return True, (
            f"OK ✅ dataset.yaml válido\n"
            f"dataset.yaml: {y}\n"
            f"Dataset root resuelto: {root}\n"
            f"{names_note}"
        )

    except Exception as e:
        return False, f"Error leyendo/validando YAML: {e}"


# ====== Redirección de stdout a la GUI ======

class TextRedirector(io.TextIOBase):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, s):
        self.text_widget.after(0, self._append_text, s)

    def _append_text(self, s):
        self.text_widget.insert(tk.END, s)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


# ====== Clase principal de la GUI ======

class YOLOTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Entrenamiento YOLOv8 - Microplásticos (PET/PP/LDPE)")
        self.root.geometry("980x680")

        self.training_thread = None
        self.stop_training_flag = False

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(main_frame, text="Parámetros de entrenamiento", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Modelo
        ttk.Label(left_frame, text="Modelo YOLOv8:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="yolov8n.pt")
        ttk.Combobox(
            left_frame,
            textvariable=self.model_var,
            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            state="readonly",
            width=15,
        ).grid(row=0, column=1, sticky="w", pady=2)

        # Imagen size
        ttk.Label(left_frame, text="Tamaño de imagen:").grid(row=1, column=0, sticky="w")
        self.img_size_var = tk.IntVar(value=640)
        ttk.Entry(left_frame, textvariable=self.img_size_var, width=10).grid(row=1, column=1, sticky="w", pady=2)

        # Epochs
        ttk.Label(left_frame, text="Epochs:").grid(row=2, column=0, sticky="w")
        self.epochs_var = tk.IntVar(value=100)
        ttk.Entry(left_frame, textvariable=self.epochs_var, width=10).grid(row=2, column=1, sticky="w", pady=2)

        # Batch size
        ttk.Label(left_frame, text="Batch size:").grid(row=3, column=0, sticky="w")
        self.batch_var = tk.IntVar(value=16)
        ttk.Entry(left_frame, textvariable=self.batch_var, width=10).grid(row=3, column=1, sticky="w", pady=2)

        # Learning rate
        ttk.Label(left_frame, text="Learning rate (lr0):").grid(row=4, column=0, sticky="w")
        self.lr_var = tk.DoubleVar(value=0.01)
        ttk.Entry(left_frame, textvariable=self.lr_var, width=10).grid(row=4, column=1, sticky="w", pady=2)

        # Aumentación
        ttk.Label(left_frame, text="Aumento de datos:").grid(row=5, column=0, sticky="w")
        self.aug_var = tk.StringVar(value="Medio (recomendado)")
        ttk.Combobox(
            left_frame,
            textvariable=self.aug_var,
            values=["Sin aumento extra", "Suave", "Medio (recomendado)", "Fuerte"],
            state="readonly",
            width=20,
        ).grid(row=5, column=1, sticky="w", pady=2)

        # dataset.yaml
        ttk.Label(left_frame, text="dataset.yaml:").grid(row=6, column=0, sticky="w")
        self.data_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.data_var, width=38).grid(row=6, column=1, sticky="w", pady=2)
        ttk.Button(left_frame, text="Buscar...", command=self.browse_data_yaml).grid(row=6, column=2, padx=5)

        # Carpeta proyecto (runs)
        ttk.Label(left_frame, text="Carpeta proyecto:").grid(row=7, column=0, sticky="w")
        self.project_var = tk.StringVar(value="runs_microplastico")
        ttk.Entry(left_frame, textvariable=self.project_var, width=38).grid(row=7, column=1, sticky="w", pady=2)
        ttk.Button(left_frame, text="Elegir...", command=self.browse_project_dir).grid(row=7, column=2, padx=5)

        # Nombre experimento
        ttk.Label(left_frame, text="Nombre experimento:").grid(row=8, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.name_var, width=38).grid(row=8, column=1, sticky="w", pady=2)
        ttk.Button(left_frame, text="Sugerir", command=self.suggest_name).grid(row=8, column=2, padx=5)

        # GPU + memoria
        self.gpu_label = ttk.Label(left_frame, text="GPU: (no detectada)")
        self.gpu_label.grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 2))

        self.mem_label = ttk.Label(left_frame, text="Memoria estimada: -")
        self.mem_label.grid(row=10, column=0, columnspan=3, sticky="w")

        ttk.Button(left_frame, text="Calcular memoria", command=self.calculate_memory).grid(
            row=11, column=0, columnspan=3, pady=5, sticky="we"
        )

        ttk.Button(left_frame, text="Iniciar entrenamiento", command=self.start_training).grid(
            row=12, column=0, columnspan=3, pady=(15, 5), sticky="we"
        )
        ttk.Button(left_frame, text="Detener (forzar)", command=self.stop_training).grid(
            row=13, column=0, columnspan=3, pady=5, sticky="we"
        )

        right_frame = ttk.LabelFrame(main_frame, text="Log de entrenamiento", padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(right_frame, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr
        sys.stdout = TextRedirector(self.log_text)
        sys.stderr = TextRedirector(self.log_text)

        self.update_gpu_label()

        footer = ttk.Label(self.root, text="Cristofher Ferrada — Microplásticos 2026", anchor="center")
        footer.pack(side=tk.BOTTOM, pady=5)

    def browse_data_yaml(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar dataset.yaml", filetypes=[("YAML files", "*.yaml *.yml"), ("Todos", "*.*")]
        )
        if filename:
            self.data_var.set(filename)

    def browse_project_dir(self):
        dirname = filedialog.askdirectory(title="Seleccionar carpeta de proyecto (runs)")
        if dirname:
            self.project_var.set(dirname)

    def suggest_name(self):
        model = self.model_var.get().replace(".pt", "")
        img = self.img_size_var.get()
        batch = self.batch_var.get()
        epochs = self.epochs_var.get()
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        suggested = f"MP_{model}_img{img}_b{batch}_e{epochs}_{now}"
        self.name_var.set(suggested)

    def update_gpu_label(self):
        info = get_gpu_info()
        if info is None:
            self.gpu_label.config(text="GPU: no se detectó CUDA (entrenando en CPU)")
        else:
            self.gpu_label.config(
                text=f"GPU: {info['name']} | VRAM ~ {info['total_gb']:.1f} GB | device={info['device_index']}"
            )

    def calculate_memory(self):
        try:
            model_name = self.model_var.get()
            img = int(self.img_size_var.get())
            batch = int(self.batch_var.get())
        except ValueError:
            messagebox.showerror("Error", "Revisa que imagen y batch sean números válidos.")
            return

        info = get_gpu_info()
        est = estimate_memory_gb(model_name, img, batch)

        if info is None:
            msg = f"Memoria estimada requerida: ~{est:.1f} GB\nNo se detectó GPU, se usará CPU (muy lento)."
        else:
            total = info["total_gb"]
            ratio = est / total
            if ratio < 0.6:
                estado = "✅ Debería aguantar."
            elif ratio < 0.9:
                estado = "⚠️ Ajustado: cerca del límite."
            else:
                estado = "❌ Probable OOM: baja batch o img."
            msg = (
                f"Modelo: {model_name}\nImagen: {img}x{img}\nBatch: {batch}\n"
                f"Memoria estimada: ~{est:.1f} GB\nVRAM GPU: ~{total:.1f} GB\n\n{estado}"
            )

        self.mem_label.config(text=msg.replace("\n", " | "))
        messagebox.showinfo("Estimación de memoria", msg)

    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Entrenamiento en curso", "Ya hay un entrenamiento en ejecución.")
            return

        data_path = self.data_var.get().strip()
        if not os.path.isfile(data_path):
            messagebox.showerror("Error", "Debes seleccionar un dataset.yaml válido.")
            return

        # Validación dura del YAML y rutas reales
        ok, msg = validate_dataset_yaml(data_path)
        if not ok:
            messagebox.showerror("Dataset inválido", msg)
            return
        else:
            print(msg)

        # Rutas absolutas para evitar confusión
        data_path = os.path.abspath(data_path)

        project_dir = self.project_var.get().strip()
        if not project_dir:
            messagebox.showerror("Error", "Debes indicar una carpeta de proyecto.")
            return
        project_dir = os.path.abspath(project_dir)

        run_name = self.name_var.get().strip()
        if not run_name:
            self.suggest_name()
            run_name = self.name_var.get().strip()

        try:
            img = int(self.img_size_var.get())
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            lr0 = float(self.lr_var.get())
        except ValueError:
            messagebox.showerror("Error", "Revisa que imagen, epochs, batch y lr sean válidos.")
            return

        model_name = self.model_var.get()
        aug_level = self.aug_var.get()

        self.log_text.insert(tk.END, "\n========== INICIANDO ENTRENAMIENTO ==========\n")
        self.log_text.see(tk.END)

        args = {
            "model_name": model_name,
            "data_path": data_path,
            "img": img,
            "epochs": epochs,
            "batch": batch,
            "lr0": lr0,
            "project_dir": project_dir,
            "run_name": run_name,
            "aug_level": aug_level,
        }

        self.training_thread = threading.Thread(target=self._train_worker, args=(args,))
        self.training_thread.daemon = True
        self.training_thread.start()

    def _ask_save_trained_model(self, default_weights_path: str):
        if not os.path.isfile(default_weights_path):
            messagebox.showwarning("Guardar modelo", f"No se encontró:\n{default_weights_path}")
            return

        resp = messagebox.askyesno(
            "Guardar modelo",
            "El entrenamiento terminó.\n¿Quieres guardar una copia del best.pt en otra ubicación?",
        )
        if not resp:
            return

        dest = filedialog.asksaveasfilename(
            title="Guardar modelo entrenado",
            defaultextension=".pt",
            initialfile=os.path.basename(default_weights_path),
            filetypes=[("PyTorch weights", "*.pt"), ("Todos", "*.*")],
        )
        if not dest:
            return

        try:
            shutil.copy2(default_weights_path, dest)
            messagebox.showinfo("Guardar modelo", f"Modelo guardado en:\n{dest}")
        except Exception as e:
            messagebox.showerror("Error al guardar modelo", str(e))

    def _train_worker(self, args):
        try:
            model_name = args["model_name"]
            data_path = args["data_path"]
            img = args["img"]
            epochs = args["epochs"]
            batch = args["batch"]
            lr0 = args["lr0"]
            project_dir = args["project_dir"]
            run_name = args["run_name"]
            aug_level = args.get("aug_level", "Medio (recomendado)")

            device = 0 if torch.cuda.is_available() else "cpu"

            print(f"\nYAML usado: {data_path}")
            print(f"Root resuelto (según YAML): {resolve_dataset_root(data_path)}\n")

            print(f"Cargando modelo {model_name} ...")
            model = YOLO(model_name)

            aug_params = get_augmentation_params(aug_level)
            print(f"Augmentación: {aug_level} -> {aug_params}\n")
            print(f"Device: {device}\n")
            print(f"Project dir: {project_dir}")
            print(f"Run name: {run_name}\n")

            model.train(
                data=data_path,
                imgsz=img,
                epochs=epochs,
                batch=batch,
                lr0=lr0,
                project=project_dir,
                name=run_name,
                device=device,
                workers=4,
                **aug_params,
            )

            print("\n===== ENTRENAMIENTO FINALIZADO =====")
            run_dir = os.path.join(project_dir, run_name)
            best_path = os.path.join(run_dir, "weights", "best.pt")
            print(f"Carpeta experimento: {run_dir}")
            print(f"best.pt: {best_path}")

            self.root.after(0, self._ask_save_trained_model, best_path)

        except Exception as e:
            print("\n[ERROR EN ENTRENAMIENTO]")
            print(str(e))

    def stop_training(self):
        self.stop_training_flag = True
        messagebox.showinfo(
            "Aviso",
            "Detener un entrenamiento interno de Ultralytics no es trivial.\n"
            "Para stop real, lo ideal es correr en un proceso separado.\n"
            "Este botón queda como placeholder.",
        )


def main():
    root = tk.Tk()
    app = YOLOTrainerGUI(root)
    root.mainloop()

    sys.stdout = app.stdout_backup
    sys.stderr = app.stderr_backup


if __name__ == "__main__":
    main()