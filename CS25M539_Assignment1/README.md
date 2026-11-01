# VGG-6 Image Classification on CIFAR-10

A modular implementation based on the `vgg6_cifar10_ImgClassification.ipynb` assessment notebook. It preserves the VGG-6 architecture, transforms, seed-controlled 45,000/5,000 train-validation split, experiment groups, and validation-based checkpoint selection while separating model, data, training/evaluation, experiments, plots, and CLI code.

---

## 1. Workspace & Environment

This repository is designed around `/content/CS25M539_Assignment1` as the **single primary workspace** when running in Google Colab (or a local equivalent). All outputs, logs, and checkpoints are stored locally inside `assessment_results/`.

### Recommended Environment
* **Python**: `3.11.x`
* **PyTorch**: `2.5.1+` (with matching torchvision)
* **Dependencies**: Pin-specified in `requirements.txt`

### Installation

#### Linux / macOS / Google Colab
```bash
git clone <repository-url> /content/CS25M539_Assignment1
cd /content/CS25M539_Assignment1
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify setup and GPU availability:
```bash
python -c "import torch, torchvision, numpy, pandas, matplotlib, wandb; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
```

---

## 1. Set up Google Colab and extract the project

1. Open [Google Colab](https://colab.research.google.com/) and create a notebook.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU). GPU is recommended for training.
3. Upload `CS25M539_Assignment1.zip`.

```python
from google.colab import files
uploaded = files.upload()  # Select CS25M539_Assignment1.zip

import zipfile
from pathlib import Path

zip_path = next(name for name in uploaded if name.lower().endswith(".zip"))
with zipfile.ZipFile(zip_path, "r") as archive:
    archive.extractall("/content")

project_dir = Path("/content/CS25M539_Assignment1")
project_dir.mkdir(parents=True, exist_ok=True)

print("Project directory:", project_dir)
print("main.py:", (project_dir / "main.py").exists())
print("plot_from_csv.py:", (project_dir / "plot_from_csv.py").exists())
print("README.md:", (project_dir / "README.md").exists())

assert (project_dir / "main.py").exists(), "main.py was not found."
assert (project_dir / "plot_from_csv.py").exists(), "plot_from_csv.py was not found."
```

Check Python and GPU:

```python
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 2. Define the primary assessment folders

All primary outputs are kept inside `project_dir`. Do not use Google Drive paths in training or reporting commands.

```python
from pathlib import Path

project_dir = Path("/content/CS25M539_Assignment1")

data_dir = project_dir / "data"
assessment_dir = project_dir / "assessment_results"

baseline_dir = assessment_dir / "baseline"
activation_dir = assessment_dir / "activation"
optimizer_dir = assessment_dir / "optimizer"
hyperparameters_dir = assessment_dir / "hyperparameters"
all_results_dir = assessment_dir / "all"
best_rerun_dir = assessment_dir / "best_config_rerun"
wandb_reports_dir = assessment_dir / "wandb_reports"

for folder in [
    data_dir, baseline_dir, activation_dir, optimizer_dir,
    hyperparameters_dir, all_results_dir, best_rerun_dir,
    wandb_reports_dir
]:
    folder.mkdir(parents=True, exist_ok=True)

print("Primary assessment workspace:", project_dir)
print("Dataset directory:", data_dir)
print("All experiment results:", assessment_dir)
```

## 3. Expected Directory Structure

```text
/content/CS25M539_Assignment1/
    ├── main.py
    ├── plot_from_csv.py
    ├── README.md
    ├── requirements.txt
    ├── vgg6/               # Core modules (model, data, engine, experiments)
    ├── data/               # CIFAR-10 dataset (auto-downloaded)
    └── assessment_results/
        ├── baseline/
        ├── activation/
        ├── optimizer/
        ├── hyperparameters/
        ├── all/            # Complete 21-run experiment sweep (epoch_history.csv, experiment_summary.csv)
        ├── best_config_rerun/ # Independent verification run for the best configuration
        └── wandb_reports/  # Local generated plots and CSV reporting logs
```

---

## 4. Exact Execution Commands

Run all commands from the project root (`/content/CS25M539_Assignment1`). CIFAR-10 downloads automatically to `data/` on first execution.

### 4.1 Baseline Experiment
```bash
python main.py \
  --group baseline \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --activation relu \
  --optimizer adam \
  --seed 42 \
  --data-dir ./data \
  --output-dir ./assessment_results/baseline
```

### 4.2 Individual Experiment Groups
```bash
python main.py --group activation --epochs 20 --seed 42 --data-dir ./data --output-dir ./assessment_results/activation
python main.py --group optimizer --epochs 20 --seed 42 --data-dir ./data --output-dir ./assessment_results/optimizer
python main.py --group hyperparameters --seed 42 --data-dir ./data --output-dir ./assessment_results/hyperparameters
```

### 4.3 Complete Experiment Sweep (All 21 Configurations)
Runs all variations across activations, optimizers, and hyperparameters to generate the summary metrics for analysis and W&B reporting:
```bash
python main.py \
  --group all \
  --epochs 20 \
  --seed 42 \
  --data-dir ./data \
  --output-dir ./assessment_results/all
```

### 4.4 Independent Verification of Best Validation Configuration
Automatically identifies the top-performing configuration from `experiment_summary.csv` and executes a separate verification run:
```bash
python main.py \
  --group verify-best \
  --summary-csv ./assessment_results/all/experiment_summary.csv \
  --seed 42 \
  --data-dir ./data \
  --output-dir ./assessment_results/best_config_rerun
```

---

## 5. W&B Reporting from Existing CSV Files (No Retraining)

To generate W&B reports and local plots without re-running training, use `plot_from_csv.py`. It reads `epoch_history.csv` and `experiment_summary.csv` directly.

Authenticate to W&B:
```bash
wandb login
```

Run reporting and upload:
```bash
python plot_from_csv.py \
  --history-csv ./assessment_results/all/epoch_history.csv \
  --summary-csv ./assessment_results/all/experiment_summary.csv \
  --output-dir ./assessment_results/wandb_reports \
  --wandb-project vgg6-cifar10
```

### W&B Project Suffixes Created:
1. `<prefix>-curves`: One W&B run per experiment tracking loss and accuracy curves by epoch.
2. `<prefix>-scatter`: Scatter plot of validation accuracy versus step/epoch across all runs.
3. `<prefix>-configurations`: Experiment configuration table and parallel-coordinates visualization using `best_val_accuracy` as the color/metric dimension.

### Alternative Reporting Modes:
* **Local plots only:**
  ```bash
  python plot_from_csv.py --history-csv ./assessment_results/all/epoch_history.csv --summary-csv ./assessment_results/all/experiment_summary.csv --output-dir ./assessment_results/wandb_reports --local-only
  ```
* **W&B upload only (no local PNGs):**
  ```bash
  python plot_from_csv.py --history-csv ./assessment_results/all/epoch_history.csv --summary-csv ./assessment_results/all/experiment_summary.csv --output-dir ./assessment_results/wandb_reports --wandb-only --wandb-project vgg6-cifar10
  ```

---

## 6. Optional Google Drive Backup

The workspace on `/content/CS25M539_Assignment1` remains the primary location during execution. To copy completed results to Google Drive as a backup:

```python
from pathlib import Path
import shutil

project_dir = Path("/content/CS25M539_Assignment1")
backup_root = Path("/content/drive/MyDrive/CS25M539_Assignment1_backup")

try:
    from google.colab import drive
    drive.mount("/content/drive")
    if Path("/content/drive/MyDrive").exists():
        backup_root.parent.mkdir(parents=True, exist_ok=True)
        if backup_root.exists():
            shutil.rmtree(backup_root)
        shutil.copytree(project_dir, backup_root)
        print("Backup created successfully at:", backup_root)
except Exception as exc:
    print("Drive backup skipped:", exc)
```

---

## 7. Data Preprocessing & Reproducibility

* **Transforms:** `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))`, and `Cutout(length=16)` (training only).
* **Data Split:** Official 50,000 training images are deterministically split into 45,000 training and 5,000 validation samples using seed `42`. The 10,000 test images remain strictly held out.
* **Determinism:** Seed `42` initializes Python `random`, NumPy, PyTorch CPU/CUDA generators, and configures deterministic cuDNN flags.
```