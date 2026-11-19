# C3RS tabular FT-Transformer

Classify a recorded railway event using structured C3RS features only. This
adapts the `rtdl_revisiting_models.FTTransformer` setup in your NBI
`ft_transformer.ipynb`: separate continuous/categorical tensors, the library's
default architecture and optimizer, and cross-entropy or optional focal loss.
The NBI project is reference only and is not imported or modified.

## Setup and run

Install dependencies with `python -m pip install -r requirements.txt`.

Prepare one binary event target using the existing data preparation script:

```powershell
python preprocessing_data.py --target "Station Ops"
python train.py --dataset A --output checkpoints/station_ops
```

`preprocessing_data.py` reads the raw CSV configured by `DATA_PATH` in `.env`
(or `--raw`). It preserves the existing narrative-grouped 70/15/15 split and
excludes narratives, IDs, dates, event outcomes, and the listed leakage fields
from model inputs. Narratives are used only for grouping duplicate reports.
Other targets are listed by `python preprocessing_data.py --help`. Preparing
another target replaces the processed A CSVs, so use a distinct checkpoint
folder for each target.

For the focal-loss option used in the NBI notebook:

```powershell
python train.py --dataset A --loss focal --output checkpoints/station_ops_focal
```

Defaults: 30 epochs, batch size 256, patience 8, seed 42, FP32. The model uses
`FTTransformer.get_default_kwargs()` and `model.make_default_optimizer()`.
`--precision FP16` or `BF16` enables mixed precision on CUDA.

## Dataset adaptation

The three CSVs live in `datasets/processed/A/A_training.csv`,
`A_validation.csv`, and `A_test.csv`, each with a `label` column.
Counts, experience, visibility, and hours into shift are continuous inputs:
median imputation and standardization are fitted on training only.
Each remaining checkbox is its own categorical input, using the NBI notebook's
`OrdinalEncoder` and +1 shift (zero is reserved for unknown categories).
No one-hot expansion or temporal sequences are created. Time-of-day checkboxes
are ordinary report attributes, not a history of observations.

The current target is binary, so the classifier produces two logits rather
than the ten NBI deck-rating logits. The six event categories can overlap;
the current setup trains one yes/no classifier per chosen event, using the
existing preparation workflow. BERT and text fusion are future work.

Validation Macro-F1 selects `best_ft_transformer.pt`. The output folder also
contains `preprocessor.joblib` and `history.csv`. The checkpoint includes model
settings, feature order, class names, and the best epoch. The test split is
held out from training and model selection; this command does not report test
performance. Unlike the NBI notebook, this uses the existing C3RS fixed
partitions rather than bridge-grouped cross-validation.

## Files

- `train.py`: FT-Transformer training, early stopping, and checkpoint saving.
- `eval.py`: validation for continuous/categorical input batches.
- `data/`: existing CSV loading and FT input preprocessing.
- `preprocessing_data.py`: existing C3RS feature, label, and split preparation.
- `scripts/run_baselines.py`: existing baseline comparisons.
- `notebooks/`: existing exploratory analysis and target review.
