# Machine Learning for Human Factors Analysis in Railway Safety Reports

This repository trains a fully connected neural network (MLP) for multiclass
classification on tabular CSV data. It includes CSV loading, train-only feature
preprocessing, model training, and validation metrics.

## Requirements

- Python 3.10 or newer
- pip
- PyTorch, installed for your platform (CPU or CUDA)

From the project root, create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn tqdm
```

Install PyTorch separately using the command recommended by the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
The code uses CUDA when available and otherwise runs on CPU.

## Dataset format

The only dataset choice currently configured is `A`. Place its CSV partitions
in `datasets/processed/A/` with these exact names:

```text
datasets/
└── processed/
    └── A/
        ├── A_training.csv
        ├── A_validation.csv
        └── A_test.csv
```

Each CSV must have a header row, at least one feature column, and a target
column named `label`. Column names are stripped of surrounding whitespace and
normalized to lowercase when loaded. Each partition must contain data, and
`label` cannot contain missing values. The training and validation feature
columns must match in both names and order.

Numeric and boolean features are median-imputed and standardized using
Z-score scaling. Other feature columns are imputed with their most frequent
value and one-hot encoded. Preprocessing is fitted only on the training
partition and reused for validation.

The repository currently includes small synthetic CSVs for trying the
pipeline. Replace them with your own partitions when ready. The test partition
can be loaded with `load_processed_split("A", "test")`; it is not used during
the training or validation loop.

## Train

From the project root:

```powershell
python train.py --dataset A
```

The default run uses 10 epochs, batch size 3, learning rate `0.0001`, SGD
momentum `0.5`, seed 42, and FP32 precision. For a quicker smoke test:

```powershell
python train.py --dataset A --epochs 1 --batch-size 128
```

Supported options:

| Option | Default | Description |
| --- | ---: | --- |
| `--dataset` | Required | Dataset choice; currently only `A` |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size`, `-b` | `3` | Samples per batch |
| `--num-workers`, `-j` | `0` | DataLoader worker processes |
| `--precision` | `FP32` | `FP32`, `FP16`, or `BF16`; AMP is used only on CUDA |
| `--seed` | `42` | Random seed |
| `--lr` | `0.0001` | SGD learning rate |
| `--momentum` | `0.5` | SGD momentum |
| `--output` | `checkpoints` | Reserved output directory argument |

Training prints loss and Macro-F1 for training and validation after each epoch.
The current training script does not yet save model checkpoints.

## Repository layout

```text
data/
  data_loading.py       CSV partition loading and label encoding
  preprocessing.py      Imputation, categorical encoding, and Z-score scaling
datasets/
  raw/                  Source datasets
  processed/A/          Processed training, validation, and test CSVs
models/
  MLP.py                Fully connected classifier
utilis/
  loss.py               Cross-entropy loss
  metrics.py            Macro-F1 metric
eval.py                 Validation loop and metrics
train.py                CLI training entry point
preprocessing_data.py   Placeholder for the one-time data preparation script
```

## Current limitations

- Only dataset choice `A` is enabled.
- `preprocessing_data.py` is currently a placeholder; prepare the processed
  CSV partitions separately.
- Dataset files are excluded from Git by `.gitignore`, so add or copy them
  locally before running the training command.
- There is no standalone test evaluation command or checkpoint saving yet.
