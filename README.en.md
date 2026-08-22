# Poly-X — Microplastics detection suite

*[Español](README.md) · **English***

> **Automated detection and classification of microplastics (PET, PP, LDPE)
> by Nile Red fluorescence under UV light (254 nm), using YOLO v8/v11 models.**

**Author:** Cristofher Ferrada · PhD in Chemistry · PUCV · 2026
**Version:** 2.0.0 · Windows 10/11 · Python 3.11

---

## Modules

| # | Module | Status | What it does |
|---|---|---|---|
| 1 | 🏠 **Launcher** | ✅ Working | Main menu — opens each module in its own window |
| 2 | 🔬 **Detector** | ✅ Working | Batch analysis with YOLO · paper-quality HTML report |
| 3 | 🎯 **Trainer** | ✅ Working | YOLO v8/v11 training with live curves |
| 4 | 🏷 **Labeller** | ✅ Working | Interactive YOLO annotation with automatic pre-labelling |
| 5 | 📐 **Viewer** | ✅ Working | Single-image inspection, calibration and particle-by-particle review |

---

## Features

### 🔬 Detector (batch analysis)
- Loads up to **three .pt models** at once for side-by-side comparison
- Runs over folders of images with a progress bar
- Error analysis: **true positives / false positives / false negatives /
  misclassified**, confusion matrix, gallery of the worst cases
- **Automatic calibration against the Petri dish**: the rim is found on its own
  and its known diameter sets the µm/px **of each photo**, with nothing to mark
  by hand. This matters because the shooting distance varies between takes: in
  this study's material the real scale ranges from 31 to 50 µm/px, a factor of
  1.6, so a single value for the whole batch would give sizes off by as much as
  50 %
- **Size and shape measured on the particle, not on its box**: length, width,
  area, aspect ratio and **fibre / fragment** classification. The box of an
  elongated particle is nearly empty and depends on how it happened to land — a
  fibre lying diagonally has a square box — so across 7,129 annotated particles
  it overestimated area by **1.87×**
- **Size distribution histogram** per class (PET/PP/LDPE) and by size band,
  stacked by polymer
- **CSV export** straight from the results page
- **Drag and drop** for `.pt` models and images or folders
- **Adjustable inference resolution** with presets (Fast 1280 · Balanced 2560 ·
  Maximum detection 4096) and a button that measures what your GPU can take
- Self-contained HTML report (images embedded as **base64**, paper quality) with
  a **side-by-side prediction vs ground truth gallery**. Gallery images are
  re-encoded and capped in number so the file stays openable in a browser; the
  metrics still cover every image
- **PDF export** of the report in one click, ready to email
- **Selectable report sections**: eleven tick boxes and three presets (Full ·
  Short summary · Methodological). Unticking renumbers the sections **on its
  own** and adjusts the table of contents; a ticked section with no data is left
  out anyway
- **Selectable report scope**: the whole job, only the photos you tick, or both
  at once. Figures, charts and the confusion matrix are recomputed over what you
  chose, so the report always describes the photos it shows
- **Real model comparison** inside the report: a photo-by-photo table with each
  model's detections and, when ground truth exists, their TP/FP/FN and overall F1
- **Automatic tiling of large photographs**: above a threshold the photo is
  analysed in overlapping tiles, because at full resolution the particles fall
  below the network's stride and vanish. Boxes are mapped back to the original
  photo's coordinates and overlaps merged with NMS, so **results and report are
  always delivered on the whole photo, never on the tiles**
- The Run tab warns beforehand whether the batch will be tiled and how long it
  will take

### 🎯 Trainer
- Supports **YOLO v8 and v11**, sizes nano → xlarge
- Loads `data.yaml` with automatic train/val/test split detection
- **Dataset validation** with ✓/✗ indicators (images, labels, classes)
- Live loss curves (box loss, mAP, precision, recall)
- Configurable augmentation (flip, rotation, mosaic, HSV)
- Export to **ONNX / TensorRT / CoreML**

### 🏷 Labeller
- Manual YOLO annotation with optional automatic pre-labelling
- Progress is recovered from disk, so a count can be spread over several sessions
- A `.txt` is written **only** when the image is marked as reviewed or a box is
  drawn: recording a barely glanced image as "reviewed with zero" would falsify
  the census

### 📐 Viewer
- Opens a single image, or steps through a folder with `← →`
- **Interactive µm/pixel calibration** in two modes:
  - 📏 **Line** (2 clicks): mark a known reference, type its real size
  - ⭕ **Circle** (3 clicks): mark 3 points on a circular edge, type the real
    diameter (handy with Petri dishes, whose diameter is known)
- The status bar shows `📐 0.4880 µm/px (line)` live
- Detection with a loaded `.pt` model, with **configurable inference resolution**
  (320–8192) and GPU when available. With tiny objects in high-resolution photos
  `imgsz` is decisive: at low values the particles fall below the network's
  stride and nothing is detected
- **Load `.txt` labels**: shows existing annotations over the image with their
  sizes converted to µm. Useful for reviewing a manual count without going back
  to the Labeller
- **Particle-by-particle review**: the table lists every particle with its
  **number**, class, type (fibre or fragment), length, width and aspect ratio.
  Selecting a row shows **what it was measured on**: the crop without marks on
  the left and, on the right, the mask outline with the measurement drawn over it
  — the Feret line in yellow, the geodesic path in magenta — plus the full
  pixels-to-micrometres arithmetic. A size you cannot see being measured cannot
  be verified
- **Load predictions from a closed run** in `runs/detect_.../`, without running
  the model again. It always opens the original photo and never the annotated
  PNG, which has the boxes painted onto it
- **Drag and drop** an image or a `.pt` model straight onto the canvas
- Exports: annotated image + `detecciones.csv` + `resumen.json`

---

## How a particle's size is measured

The general criterion is **the longest straight line that fits inside the
particle**, that is the greatest distance between two points of its outline: the
*maximum Feret diameter*. It does not depend on the orientation the particle
happened to land in, and a jagged edge does not disturb it.

That straight line stops working when the particle is **contorted**: in a bent
fibre the distance between the ends is the chord, and on a half-circle arc it
falls 35 % short. For those cases the *geodesic diameter* is measured instead —
the longest path that fits **inside** the particle, which cannot leave the mask
and therefore follows the curve.

| Particle shape | What is reported as length |
|---|---|
| Compact or irregular, but not bent | Maximum Feret — the longest straight line |
| Elongated and contorted (fibre) | Geodesic diameter — follows the curve |

The geodesic is only used when the particle is **thin** (length ≥ 4 × thickness)
and **non-convex** (solidity < 0.90). Without the first condition, any concavity
makes the path go around the particle instead of through it; without the second,
the length would start depending on the rotation angle, which is precisely the
defect this was meant to remove.

Against synthetic shapes of known size — straight bars, rotated bars, arcs of
60°, 120° and 180°, a circle, a bar with a jagged edge and a blob with a notch —
the length measured this way gives a **median error of 0.6 % and 4.7 % in the
worst case**. This is pinned down in `tests/test_morfologia.py`.

> **The equivalent rectangle is not a size.** The formula
> *L* = (*P* + √(*P*²−16*A*))/4 gives the length of a rectangle with the same
> area and the same perimeter, which is a different thing. It depends on the
> perimeter, so a jagged edge inflates it by 22.5 %, and it is undefined for
> compact particles, where *P*² < 16*A*. It is reported as a descriptor, because
> compared against the other two it exposes irregular outlines, but it is not
> used as a size.

**Touching particles.** Two particles in contact form a single blob, and
measuring them together would add their sizes up. They are separated by
*watershed* on the distance transform: the centre of each one sits far from the
background while the neck joining them sits close, so the cut falls on the neck.
Against circles of known size it separates them up to **27 % overlap of the
diameter**, without splitting any particle that is a single piece.

**Stated limitations.** Two particles overlapping by more than 40 % of their
diameter are still measured as one: past that point there is no neck left to cut
along. And in a tightly coiled fibre the geodesic path cuts the corner at each
bend, underestimating by up to 19 % in the tightest case tested.

---

## Polymer classes

| ID | Class | Observed fluorescence (Nile Red, UV) | Interface colour |
|---|---|---|---|
| 0 | **PET** | Red–salmon | 🔴 `#e3342f` |
| 1 | **PP** | **Dull yellow-green** | 🟠 `#ff8c00` |
| 2 | **LDPE** | Bright, clear yellow | 🟡 `#ffd700` |

The interface colours are there to tell the boxes apart on screen and **do not
describe the real emission**. Measured inside the training-set boxes (mean RGB,
n = 30 per class): PET 116/58/65 · PP 122/125/32 · LDPE 181/162/57.

**PP and LDPE share a hue and separate by brightness** (R 122 against 181), plus
PP's greenish cast. That is the dominant confusion when annotating, and it
explains why per-class recall collapses there: PET 0.98 · PP 0.70 · LDPE 0.54.

---

## Language

The interface is available in **Spanish and English**. The selector sits in the
Launcher's top bar and the choice is remembered between sessions. The first time,
the system language is used; `POLYX_IDIOMA=en` forces it without touching the
interface.

Modules are separate processes and read the language when they open, so the
change takes effect as soon as you open the next module.

To see what is still untranslated:

```bat
.venv\Scripts\python.exe auditar_traduccion.py
```

---

## Requirements

| Component | Version |
|---|---|
| Windows | 10 / 11 |
| Python | **3.11.x** (not 3.12+) |
| RAM | 8 GB minimum |
| GPU | NVIDIA optional (20–30× faster for training). The installer picks the CUDA build that matches the card's architecture |

Main dependencies: `PySide6 6.7`, `Ultralytics 8.3`, `OpenCV 4.10`, `NumPy 1.26`,
`Matplotlib 3.9`

---

## Installation

Clone or download the repository, then run once:

```bat
SETUP.bat
```

It creates `.venv`, detects the GPU and installs PyTorch, Ultralytics and
PySide6. Nothing else has to be installed by hand.

---

## Usage

Double-click the **Poly-X** shortcut on the Desktop, or:

```bat
iniciar_polyx.bat
```

That opens the **Launcher** → pick a module. Or straight from a terminal:

```bash
.venv\Scripts\python.exe -m polyx.launcher
.venv\Scripts\python.exe -m polyx.detector
.venv\Scripts\python.exe -m polyx.trainer
.venv\Scripts\python.exe -m polyx.etiquetador
.venv\Scripts\python.exe -m polyx.visor
```

---

## Updating

To pull the latest version published on GitHub **without reinstalling anything**,
double-click:

```bat
actualizar.bat
```

It checks whether there is a new commit on `main`; if there is, it downloads and
replaces only the program files. It **keeps** your `.venv` environment, your
`models\*.pt`, your `runs\` and any local data. Git does not need to be installed
(it downloads over HTTPS).

---

## Project layout

```
polyx/
├── launcher.py          # Main menu
├── core/                # Shared modules (theme, yolo_wrap, metrics, report_html,
│                        #   calibracion, morfologia, procedencia, i18n)
├── detector/            # Module 2: batch analysis (9 pages)
├── trainer/             # Module 3: YOLO training (9 pages)
├── etiquetador/         # Module 4: interactive annotation
└── visor/               # Module 5: inspection + µm/px calibration
models/                  # Trained .pt weights
runs/                    # Results of each run
data_microplastico/      # YOLO dataset (images/ + labels/)
tests/                   # Test suite for measurement and calibration
```

---

## The full workflow

```
Microscope images (UV 254 nm, Nile Red staining)
        ↓
  🏷 Labeller  → annotate PET/PP/LDPE in YOLO format
        ↓
  🎯 Trainer   → train a YOLO v8/v11 model
        ↓
  🔬 Detector  → batch analysis + HTML report
        ↓
  📐 Viewer    → detailed inspection + µm/px calibration
```

---

## Tests

Shape measurement and calibration have their own suite, because every figure they
produce ends up in a table of the paper and a well-meant change can shift them
all without anything warning you.

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
```

43 tests against **synthetic shapes of known size**, depending on no human
annotation whatsoever. Each one also pins down the *why* of a design decision, so
that if someone tries a variant that was already discarded, the suite says so.

`pytest` is only needed for development and is **not in `requirements.txt`**: an
installation meant for use does not need it.

---

## Related publications

- **Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).** Development
  of a new methodology for the determination of PET microplastics in sediment,
  based on microwave-assisted acid digestion. *PLoS ONE* 19(12): e0314520.
  https://doi.org/10.1371/journal.pone.0314520
- **Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).**
  Evaluation of microwave-assisted acid/oxidant digestion method for the detection
  of polyethylene microplastics in *Merluccius gayi* fish by Nile Red fluorescent
  staining and image analysis. *J. Chil. Chem. Soc.* 69(1): 6082-6085.
  https://doi.org/10.4067/s0717-97072024000106082

---

## User manual

`Manual_PolyX.en.html` ships with the repository: every tab documented with
screenshots. Regenerate it with:

```bash
POLYX_IDIOMA=en .venv\Scripts\python.exe generar_manual.py --manual Manual_PolyX.en.html
```

---

## Repository scope

This repository documents **the program**. Everything belonging to a paper in
preparation — the study's analysis pipeline, its photographs and its findings —
is deliberately left out, because publishing it here would pre-empt results that
are not out yet.

---

## Licence and contact

**Cristofher Ferrada** — PhD in Chemistry, Pontificia Universidad Católica de
Valparaíso, 2026.

Repository: https://github.com/CrissFerrada/Poly-X-Microplastics
