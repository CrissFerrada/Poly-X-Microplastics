# Poly-X — Microplastics detection suite

*[Español](README.md) · **English***

> **Automated detection and classification of microplastics (PET, PP, LDPE)
> by Nile Red fluorescence under UV light (254 nm), using YOLO v8/v11 models.**

**Author:** Cristofher Ferrada · PhD in Chemistry · Pontificia Universidad Católica de Valparaíso · 2026
**Version:** 2.0.0 · Windows 10/11 · Python 3.11

---

## What this is

A desktop application that covers the whole workflow of counting microplastics in
UV-fluorescence photographs: **train → label → detect → report**. It exists
because counting particles by hand in photographs of sediment is the rate-limiting
step of this kind of study, and because the counting has to stay auditable — every
module records what it did, not just its result.

| # | Module | Status | What it does |
|---|---|---|---|
| 1 | 🏠 **Launcher** | ✅ Working | Main menu; opens each module in its own window |
| 2 | 🔬 **Detector** | ✅ Working | Batch inference with YOLO · paper-quality HTML report |
| 3 | 🎯 **Trainer** | ✅ Working | YOLO v8/v11 training with live curves |
| 4 | 🏷 **Labeler** | ✅ Working | Interactive YOLO annotation with model pre-annotation |
| 5 | 📐 **Viewer** | ✅ Working | Single-image inspection with interactive µm/pixel calibration |

The interface is available in **Spanish and English** (selector in the Launcher's
top bar, or the `POLYX_IDIOMA=en` environment variable). Translation coverage is
reported by `python auditar_traduccion.py`; at the time of writing the Launcher is
fully translated and the four modules are in progress.

---

## Two design decisions worth knowing about

### Automatic tiling for large photographs

A whole-plate photograph is ~4096 px on its long side. Feeding it to the network
at `imgsz=2080` rescales it, which halves every particle: at ~12–19 px native, a
microplastic particle drops below the network's stride and simply stops existing.
The historical workaround was to cut each plate into pieces by hand and run
detection piece by piece.

The Detector, Labeler and Viewer now decide on their own. `politica_troceado()`
compares the **longest side** against a threshold — what destroys the particle is
the rescaling, not the file size — and if the photo is over it, the image is split
into overlapping tiles, each inferred at native resolution, with the boxes mapped
back to full-image coordinates and merged by global NMS.

Measured on one whole plate from the study (3072×4096): **123 detections in a
single pass versus 328 tiled** into 6 tiles of 2000 px (manual count for that
plate: 424). The overlap is what removes the seam problem — a particle straddling
two tiles is merged rather than double-counted or lost.

The default threshold of 2000 px lets the study's own 1630-px crops through in one
piece and tiles the full plates. All of it is configurable in *Parameters →
Automatic tiling*.

### The Trainer prioritizes resolution over batch size

For particles of ~12–19 px, what makes them detectable is resolution, not batch
size. So `recomendar_config()` applies a fixed order:

1. **imgsz as high as it fits** in VRAM at the smallest usable batch, capped by the
   dataset's native resolution — training at 4096 on 1630-px crops only
   interpolates and eats the VRAM the batch needs.
2. **batch** afterwards, with imgsz already fixed, using whatever is left.
3. **speed** last: AMP, workers, cache. AMP stays on permanently, because it is
   not only speed — it is what makes the high imgsz possible, so switching it off
   would violate priority 1.

GPU detection picks the card with the most VRAM rather than device 0 (on a machine
with integrated and discrete graphics, device 0 can be the wrong one), and reports
compute capability, driver and the CUDA version PyTorch was built against. If
PyTorch cannot see CUDA but `nvidia-smi` can see the card, that specific situation
is reported as such — it is an installation problem, not missing hardware, and the
old "no GPU detected" message sent people looking in the wrong place.

---

## Polymer classes

| ID | Class | Observed fluorescence (Nile Red, UV) | On-screen box color |
|---|---|---|---|
| 0 | **PET** | Red–salmon | 🔴 `#e3342f` |
| 1 | **PP** | **Dull yellow-green** | 🟠 `#ff8c00` |
| 2 | **LDPE** | Clear yellow, **brighter** | 🟡 `#ffd700` |

The colors on the right are only how boxes are drawn; they do not describe the real
emission. Mean RGB measured inside the training-set boxes (n=30 per class):

| Class | R | G | B |
|---|---|---|---|
| PET | 116 | 58 | 65 |
| PP | 122 | **125** | 32 |
| LDPE | 181 | 162 | 57 |

**PP and LDPE do not separate by hue but by brightness**: both are yellowish, but
in PP the green channel equals or exceeds the red and the emission is markedly
duller. This is the most common annotation confusion and the reason per-class
recall drops on those two.

> Fluorescence-based class assignment is **not** a chemical identification. Any
> claim about polymer composition needs FTIR or Raman confirmation on a subsample,
> and needs correcting for the per-class recall.

---

## Requirements

| Component | Version |
|---|---|
| Windows | 10 / 11 |
| Python | **3.11.x** (not 3.12+) |
| RAM | 8 GB minimum |
| GPU | NVIDIA optional (20–30× faster for training) |

Main dependencies: `PySide6 6.7`, `Ultralytics 8.3`, `OpenCV 4.10`, `NumPy 1.26`,
`Matplotlib 3.9`, `psutil 6.0`.

---

## Installation

Download the project (**Code → Download ZIP**, then unzip; or `git clone`) and
double-click:

```bat
SETUP.bat
```

The installer:

1. Asks **where to install** (press ENTER to install in place).
2. Finds Python 3.11 and creates the `.venv` environment.
3. Detects an NVIDIA GPU and downloads the matching PyTorch build (CUDA or CPU).
4. Installs the remaining dependencies and verifies that everything imports.
5. Creates `models\` and `data\`, and puts a **"Poly-X" shortcut on the Desktop**
   pointing at `Poly-X.vbs` with the application icon.

Then launch from the Desktop shortcut, or double-click `Poly-X.vbs` (starts with
`pythonw`, so no console window). `iniciar_polyx.bat` does the same but keeps the
console visible, which is useful when something fails at startup.

To pull the latest published version without reinstalling, double-click
`actualizar.bat`.

### What is *not* in the download

Two things are deliberately excluded because of their size, and have to be supplied
separately:

| What | Where it goes |
|---|---|
| A trained model, `*.pt` | drop it in `models\` |
| The training dataset | unzip anywhere, then pick its `dataset.yaml` in the Trainer |

Command-line entry points, if preferred:

```bash
.venv\Scripts\python.exe -m polyx.launcher
.venv\Scripts\python.exe -m polyx.detector
.venv\Scripts\python.exe -m polyx.trainer
.venv\Scripts\python.exe -m polyx.etiquetador
.venv\Scripts\python.exe -m polyx.visor
```

---

## Repository scope

This repository contains **the program only**. Trained weights, image datasets and
the analysis pipeline and manuscript of the study in preparation are excluded by
`.gitignore` — publishing them here would pre-empt unpublished results, the
author's and other people's.

---

## Related publications

- **Pérez M, Parra S, Ferrada C, et al.** (2024). *PLoS ONE* 19(12): e0314520.
  https://doi.org/10.1371/journal.pone.0314520
- **Ferrada C, Pérez M, Parra S, et al.** (2024). *J. Chil. Chem. Soc.* 69(1): 6082.

## Contact

Cristofher Ferrada — PhD in Chemistry, Pontificia Universidad Católica de
Valparaíso, Chile.
