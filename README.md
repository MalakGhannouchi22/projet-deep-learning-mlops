# Urban Skin

Urban Skin is a local-first computer-vision and MLOps project for detecting
infrastructure damage and turning visual evidence into an explainable condition
score and intervention priority.

## Project progress

The project has been initialized as a complete local-first computer-vision
prototype. The main work completed so far includes:

- organizing the infrastructure-damage datasets and adding audit/preparation
  scripts for reproducible data processing;
- implementing a small U-Net segmentation baseline with training and evaluation;
- adding health-score and intervention-priority logic based on predicted damage;
- exposing model metadata, health checks, and prediction endpoints through FastAPI;
- creating a React frontend for interacting with the API;
- configuring tests, Ruff, Docker Compose, DVC, MLflow, and GitHub Actions for
  development and MLOps workflows.

The current model checkpoint is a local PyTorch artifact. The API is ready to
run, but production inference requires a supplied model artifact and configured
model path.

## Current project status

The first vertical slice is intentionally honest and reproducible:

- `data/` contains independent dataset folders:
  - `data/dacl10k`: semantic segmentation, 8,922 images.
  - `data/rdd2022-DatasetNinja`: road-damage detection.
  - `data/Building defect on walls.v4i.yolov8`: wall-defect detection.
  - `data/building-facade-segmentation-original.v6i.coco-segmentation`: facade
    segmentation in COCO format.
  - `data/Corrosion Condition State Semantic Segmentation Dataset`: corrosion
    condition segmentation.
- `src/urban_skin/data/audit.py` computes split and annotation statistics without
  loading all images into memory.
- `src/urban_skin/data/prepare.py` converts DACL10K's FiftyOne JSON export into a
  compact, reproducible polygon manifest.
- `src/urban_skin/models/` contains a small U-Net segmentation baseline, training,
  and IoU/Dice evaluation for every normalized segmentation manifest.
- `src/urban_skin/decision/health.py` converts predicted damage coverage into a
  transparent condition score and intervention priority.
- `backend/app` exposes typed FastAPI contracts and `/health`, `/model/metadata`,
  and `/predict`.
- The API returns `503` until an ONNX model is supplied; it never returns fake
  predictions.
- Pytest, Ruff, DVC, Docker Compose, and GitHub Actions are wired into the project.

FloodNet and RescueNet are not installed yet. Add them later as separate folders
and dataset adapters; do not mix files between datasets.

The currently usable supervised datasets are prepared independently:

| Dataset | Local format | Records | Training treatment |
| --- | --- | ---: | --- |
| DACL10K | FiftyOne polygons | 8,923 | Native semantic segmentation |
| Wall defects | YOLO bounding boxes | 1,416 | Boxes rasterized as coarse masks |
| Building facade | COCO polygons | 2,226 | Native polygon segmentation |
| Corrosion | LabelMe polygons + class masks | 440 | Native class-mask segmentation |
| RDD2022 | DatasetNinja test split | 1,705 | Audit only; no local training split |

The wall conversion is deliberately documented as an approximation: bounding
boxes do not describe the exact defect shape. RDD2022 is not trained because
the local copy contains only `test`, so a supervised score would be invalid.

## Architecture

```text
data/<dataset> (independent folders)
              \          |          /
               -> audit + preparation -> processed/versioned data (DVC)
                                          |
                             training + MLflow experiment tracking
                                          |
                             best checkpoint -> ONNX model artifact
                                          |
                  React frontend -> FastAPI -> ONNX Runtime inference
                                          |
                      health score + graph-based priority explanation
```

Planned folders:

```text
.github/workflows/       CI checks
backend/app/             FastAPI routes, schemas, inference boundary
data/<dataset>/          one independent folder per dataset
data/processed/          generated training data, never committed
src/urban_skin/data/     dataset adapters and preparation
src/urban_skin/models/   training, evaluation, ONNX export
src/urban_skin/decision/ health score and priority graph
frontend/                React upload and visualization client
tests/                   API and data tests
models/                  local model artifacts, ignored by Git
```

## Setup in VS Code (Windows PowerShell)

Open the folder whose path is:

```text
C:\Users\Malak Ghannouchi\Documents\3eme SDIAE\projet deep learning mlops
```

In the VS Code terminal, run:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Select the interpreter with `Ctrl+Shift+P` -> **Python: Select Interpreter** ->
`.venv\Scripts\python.exe`.

## Reproducible project workflow

Run the dataset audit:

```powershell
.\.venv\Scripts\python.exe -m urban_skin.data.audit `
  --data-dir data\dacl10k `
  --output artifacts\dacl10k_audit.json
```

Prepare the segmentation manifest:

```powershell
.\.venv\Scripts\python.exe -m urban_skin.data.prepare `
  --data-dir data\dacl10k `
  --output data\processed\dacl10k_manifest.json
```

Train a quick CPU baseline. This command is intentionally small so it works
on a laptop; increase `--max-samples` and `--epochs` for a serious experiment:

```powershell
.\.venv\Scripts\python.exe -m urban_skin.models.train `
  --manifest data\processed\dacl10k_manifest.json `
  --output models\dacl10k_tiny_unet.pt `
  --max-samples 64 `
  --epochs 2 `
  --image-size 128 `
  --batch-size 4 `
  --cpu
```

Evaluate the checkpoint on validation images:

```powershell
.\.venv\Scripts\python.exe -m urban_skin.models.evaluate `
  --manifest data\processed\dacl10k_manifest.json `
  --checkpoint models\dacl10k_tiny_unet.pt `
  --max-samples 32
```

The evaluation reports `mean_iou`. This is model quality; it is not the city's
health score. The health score is calculated separately from predicted damage
coverage in `src/urban_skin/decision/health.py`.

Run checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check backend src tests
```

Start the API without weights to demonstrate readiness handling:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Open <http://127.0.0.1:8000/docs>. Demonstrate:

1. `GET /health`: the service is alive and clearly reports `model_loaded: false`.
2. `GET /model/metadata`: the typed task and class contract.
3. `POST /predict`: it returns a deliberate `503`, proving model readiness is
   checked instead of hiding a missing model.

Stop the server with `Ctrl+C`.

Start the API with the trained PyTorch baseline:

```powershell
$env:MODEL_PATH = "models\dacl10k_tiny_unet.pt"
$env:MODEL_VERSION = "dacl10k-baseline"
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

In Swagger, use `POST /predict` and upload a DACL10K image. The response
contains:

- `condition_score`: 0-100, where 100 means no predicted damage;
- `priority`: `monitor`, `planned`, or `urgent`;
- `latency_ms`: model inference time;
- `model_version`: the checkpoint identity.

Run the React dashboard in a second terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open <http://localhost:5173>, choose an image, and click **Analyze image**.
The dashboard calls the FastAPI service and displays the condition score,
priority, and latency.

The Docker serving image uses the lightweight API runtime. The local PyTorch
checkpoint is used with the project virtual environment; ONNX packaging is the
next model-serving milestone. The two containerized services can be started
together:

```powershell
docker compose up --build
```

The API is then available at <http://localhost:8000/docs> and the dashboard at
<http://localhost:5173>.

## Train the additional datasets

Generate the normalized manifests:

```powershell
.\.venv\Scripts\python.exe -m urban_skin.data.prepare_datasets walls
.\.venv\Scripts\python.exe -m urban_skin.data.prepare_datasets facade
.\.venv\Scripts\python.exe -m urban_skin.data.prepare_datasets corrosion
```

Run a small CPU smoke experiment for each dataset:

```powershell
.\.venv\Scripts\python.exe -m urban_skin.models.train_universal `
  --manifest data\processed\walls_manifest.json --dataset walls `
  --output models\walls_tiny_unet.pt --max-samples 128 --epochs 2 --cpu
.\.venv\Scripts\python.exe -m urban_skin.models.train_universal `
  --manifest data\processed\facade_manifest.json --dataset facade `
  --output models\facade_tiny_unet.pt --max-samples 128 --epochs 2 --cpu
.\.venv\Scripts\python.exe -m urban_skin.models.train_universal `
  --manifest data\processed\corrosion_manifest.json --dataset corrosion `
  --output models\corrosion_tiny_unet.pt --max-samples 128 --epochs 2 --cpu
```

Evaluate datasets that have a validation split:

```powershell
.\.venv\Scripts\python.exe -m urban_skin.models.evaluate_universal `
  --manifest data\processed\facade_manifest.json `
  --checkpoint models\facade_tiny_unet.pt
.\.venv\Scripts\python.exe -m urban_skin.models.evaluate_universal `
  --manifest data\processed\corrosion_manifest.json `
  --checkpoint models\corrosion_tiny_unet.pt
```

The evaluator reports mean IoU and Dice. These are segmentation-quality
metrics, not the explainable infrastructure health score. Walls currently have
no validation split, so they can be trained but not honestly scored until one
is created.

For the strongest local CPU experiment used for this milestone, I trained up
to 512 samples for five epochs with class-weighted cross-entropy, a 128-pixel
input, and MLflow tracking:

```powershell
.\.venv\Scripts\python.exe -m urban_skin.models.train `
  --manifest data\processed\dacl10k_manifest.json `
  --output models\dacl10k_tiny_unet_weighted.pt `
  --max-samples 512 --epochs 5 --image-size 128 --batch-size 8 --cpu `
  --mlflow --run-name dacl10k-weighted-512x5
```

The same options apply to `train_universal` for walls, facade, and corrosion.
Increasing the sample limit and epochs improves the experiment only when the
computer has enough time and memory; the command remains reproducible because
the seed is fixed.

## Current measured results

The current audit produced:

```text
8,922 images
6,935 train / 975 validation / 1,012 testdev
62,325 polygon annotations
19 damage/component classes
```

The best reproducible local CPU experiment used class-weighted cross-entropy,
512 training samples where available, five epochs, 128-pixel inputs, and
MLflow tracking. The measured outputs were:

```text
DACL10K:  final train loss 2.253571; validation mean IoU 0.000000
Walls:    final train loss 1.086029; no validation split
Facade:   final train loss 0.529286; validation mean IoU 0.000000
Corrosion: final train loss 0.561511; validation mean IoU 0.368003; Dice 0.538015
```

These results are honest local baselines, not publishable claims. The decreasing
training losses and nonzero corrosion validation score show that the full
pipeline runs, while the zero DACL10K and facade scores identify the next
research work: more data and epochs, stronger augmentation, validation
inspection, class-specific analysis, and comparison with pretrained models.
The evaluation code reports the measured result instead of hiding a weak model.

## DVC and MLflow

Do not commit the 5 GB dataset or model weights to Git. After installing the
optional ML tooling:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
dvc init
dvc add data\dacl10k
dvc repro
mlflow ui --backend-store-uri .\mlruns --port 5000
```

Training runs can be tracked directly in MLflow:

```powershell
.\.venv\Scripts\python.exe -m urban_skin.models.train `
  --manifest data\processed\dacl10k_manifest.json `
  --output models\dacl10k_mlflow.pt `
  --max-samples 64 `
  --epochs 2 `
  --mlflow `
  --run-name dacl10k-baseline
```

Then open <http://127.0.0.1:5000> to inspect parameters, loss, and checkpoint
artifacts.

The next modelling milestone is to choose one task and keep it consistent:

1. **DACL10K path:** semantic segmentation of bridge damage.
2. **RDD2022 path:** object detection of road damage.

Do not claim FloodNet, RescueNet, or ONNX results until their data and artifacts
are actually present and measured. The first PyTorch baseline is intentionally
small and should be retrained on a meaningful sample before scientific claims.

## Technical explanation of the project

I built Urban Skin as an end-to-end computer-vision and MLOps system. The
objective is to detect visible infrastructure damage, quantify the affected
area, convert that evidence into an explainable condition score, and expose
the result through an API and a web dashboard.

### Datasets

- **DACL10K** contains bridge and infrastructure images with polygon-level
  semantic-segmentation labels. It is the main dataset for the first idea.
  Each pixel is assigned to background or one of 19 damage/component classes.
- **Building defect on walls** contains YOLO bounding boxes for five wall
  problems. Its boxes are rasterized into rectangular masks so it can use the
  common segmentation trainer. This is useful for integration, but it is an
  approximation because a box is not an exact defect contour.
- **Building facade** contains COCO polygon segmentation labels for 11 facade
  and surrounding-object classes. The COCO polygons are converted into the
  normalized manifest without losing their shapes.
- **Corrosion Condition State** contains LabelMe annotations and class masks
  for fair, poor, and severe corrosion. The existing bitmap masks preserve the
  class IDs and are used directly.
- **RDD2022** is present locally for auditing, but the available copy contains
  only a test split. I do not train or report a supervised score for it without
  labeled training data.
- FloodNet and RescueNet are not included because they are not installed. They
  remain future independent adapters and are not represented as completed work.

Every dataset remains in its own folder under `data/`. The adapters create
small JSON manifests under `data/processed/`, which makes the conversion
repeatable and keeps the original files untouched.

### Models

The current baseline is a **Tiny U-Net** implemented in PyTorch. U-Net has an
encoder that learns visual features and a decoder that restores spatial
resolution, making it appropriate for pixel-level damage masks. “Tiny” means
the network is intentionally lightweight enough to train locally.

The same architecture is used separately per dataset; the datasets are not
silently mixed because their labels describe different visual concepts. The
wall model consumes approximated masks, while DACL10K, facade, and corrosion
models consume polygon or bitmap segmentation labels.

This is a defensible baseline, not a claim that Tiny U-Net is the final best
model. A production comparison should add a stronger pretrained segmentation
model after the data pipeline is validated. Detection models such as YOLO are
appropriate for the original wall and RDD2022 bounding-box tasks, but YOLO is
not the model used for the current DACL10K segmentation path.

The trainer uses deterministic seeds and inverse-frequency pixel weighting
bounded to a safe range. This reduces the tendency to predict only background,
which is a common failure mode when damage pixels are much less frequent than
background pixels. The weighting is calculated from each training manifest, not
copied between datasets.

### Metrics

- **Cross-entropy loss** is optimized during training. Lower values mean the
  predicted class distribution is closer to the target mask.
- **Intersection over Union (IoU)** is the intersection divided by the union of
  predicted and true pixels. Mean IoU averages the class IoUs and penalizes
  inaccurate regions.
- **Dice score** is `2 * intersection / (predicted area + true area)`. It is
  especially useful when damaged pixels are much rarer than background.
- **Condition score** is a separate transparent decision output from
  `src/urban_skin/decision/health.py`; it is not a substitute for IoU or Dice.
  It converts predicted damage coverage into a 0–100 score, where higher means
  less predicted damage, and maps it to monitor, planned, or urgent priority.
- **Latency** is reported by the API to measure serving responsiveness.

Validation metrics must only be reported for a real validation split. Since the
wall dataset currently has only training data, the project trains it but does
not invent a validation score.

### Technologies and their roles

- **Python and PyTorch:** data loading, model definition, optimization, and
  checkpoint creation.
- **Pillow and NumPy:** image and mask conversion.
- **FastAPI and Pydantic:** typed HTTP inference service and validation.
- **React and Vite:** upload interface and result dashboard.
- **MLflow:** records training parameters, losses, experiments, and checkpoint
  artifacts.
- **DVC:** tracks large dataset versions and reproducible data stages without
  putting gigabytes of data in Git.
- **Docker Compose:** runs the API and frontend as reproducible services.
- **Pytest and Ruff:** automated behavior tests and Python linting.
- **GitHub Actions:** continuous checks for changes pushed to the repository.
- **Git and GitHub:** source history and collaboration; model weights and raw
  datasets remain local or DVC-managed.

### How the pieces connect

1. I audit each dataset and confirm its format and available splits.
2. I run the relevant adapter to create a normalized manifest.
3. I load images and masks with the shared PyTorch dataset class.
4. I train one Tiny U-Net checkpoint per dataset and optionally log the run in
   MLflow.
5. I evaluate only datasets with validation data using IoU and Dice.
6. The API loads a supplied checkpoint, predicts a mask, measures damage
   coverage, and returns the condition score and priority.
7. React uploads an image and displays the typed API response.
8. Ruff, Pytest, DVC, the frontend build, Docker, and endpoint checks verify
   the complete workflow.

The result is a reproducible baseline: data preparation, training, evaluation,
decision logic, serving, interface, tracking, versioning, testing, and
containerization are all connected while unsupported datasets and unavailable
artifacts are explicitly reported rather than simulated.
