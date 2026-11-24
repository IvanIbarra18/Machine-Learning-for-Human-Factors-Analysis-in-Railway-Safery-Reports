# C3RS tabular FT-Transformer

Classify the six recorded railway event categories using structured C3RS
features only. One `rtdl_revisiting_models.FTTransformer` predicts all six
events at once. A report can have several events, so each event is its own
yes/no output (multi-label), not one choice among six. This adapts the setup in
your NBI `ft_transformer.ipynb`: separate continuous/categorical tensors and
the library's default architecture and optimizer. The NBI project is reference
only and is not imported or modified.

## Setup and run

Install dependencies with `python -m pip install -r requirements.txt`.

```powershell
python preprocessing_data.py
python scripts/run_baselines.py
python train.py --dataset A
```

`preprocessing_data.py` reads the raw CSV configured by `DATA_PATH` in `.env`
(or `--raw`). It preserves the existing narrative-grouped 70/15/15 split and
excludes narratives, IDs, dates, event outcomes, and the listed leakage fields
from model inputs. Narratives are used only for grouping duplicate reports.

`train.py` trains the model, then scores it on the test set. To train several
models and average their probabilities, pass several seeds:

```powershell
python train.py --dataset A --seeds 42 43 44 45 46
```

Defaults: up to 60 epochs, batch size 64, patience 8, seed 42, FP32. The model
uses `FTTransformer.get_default_kwargs()` and `model.make_default_optimizer()`
(AdamW, learning rate 1e-4). With that small learning rate the model needs
roughly 1,600 to 2,600 optimizer steps to peak: in a five-seed run at batch
size 64 the best epoch fell between 14 and 23. The earlier default of batch
size 256 for 30 epochs stopped at about 870 steps. `--loss focal` switches from binary cross-entropy to binary focal
loss. `--precision FP16` or `BF16` enables mixed precision on CUDA.

## How every model is scored

The baselines and the FT-Transformer are scored the same way, with shared code:

- Same reports: `make_split` in `preprocessing_data.py` makes the split for
  both the processed CSVs and `scripts/run_baselines.py`.
- Model choices use validation only. The baselines pick their settings by
  validation average precision (AP); `train.py` keeps the epoch with the best
  validation mean AP over the six events.
- Each event's yes/no cutoff is picked on validation, then applied to test
  (`score_event` in `utilis/metrics.py`).
- The test set is scored once per trained model: precision, recall, F1, and AP
  for each event. "Average F1" is the mean of the six per-event F1 scores.

## Results

Every experiment has its own folder under `results/`, named for what it is,
with one CSV per seed plus `summary.md`:

```
results/tabular_only/             FT-Transformer on the structured fields (train.py)
    seed_42.csv ... seed_46.csv   each model's test scores per event
    average.csv                   test scores of the averaged prediction (several seeds only)
    summary.md                    the short tables
    baselines/                    LogReg, RandomForest, XGBoost (scripts/run_baselines.py)
        seed_42.csv, summary.md
    notebook/                     the notebook's own run
        seed_42.csv, summary.md
```

Results are never overwritten. If an experiment's folder already has results,
the run writes to a new numbered folder (`results/tabular_only_2/`,
`baselines_2/`, ...) and prints which one it used. Pick the name with
`--experiment`, for example `python train.py --dataset A --experiment text_only`.

Git keeps the `summary.md` files, so the result tables are backed up. The CSVs
(`*.csv` is ignored) and the model checkpoints stay local.

## Notebook

`notebooks/fttransformer.ipynb` shows the same pipeline end to end with its
outputs saved: the data and split, the three baselines, one FT-Transformer
trained with seed 42 (loss per epoch and a training curve), the test scores of
all four models side by side, a confusion matrix per event, and the five-seed
summary read from the `train.py` result files. It imports the feature,
split, input-preparation, and scoring code from this repo, so its numbers are
comparable with `train.py` and the baselines. Choose *Run All* (about 8
minutes on the GPU), then save. It writes its results to
`results/tabular_only/notebook/`.

The test set has 1,540 reports, with about 210 to 380 "yes" reports per event.
Resampling a split of that size moves mean AP by about 0.014 (one standard
deviation), so differences of a few hundredths between models are not
meaningful.

## Text encoder comparison

`scripts/compare_encoders.py` checks which frozen text encoder to build the
text branch on, before any fine-tuning:

```powershell
python scripts/compare_encoders.py
```

For `bert-base-uncased`, `distilbert-base-uncased`, and `roberta-base` it joins
each report's narratives, cuts the text to 512 tokens, averages the frozen
encoder's token vectors into one vector per report, fits logistic regression
per event on the training reports, and reports validation AP per event. It
uses the shared split and never reads the test reports. About 5% of reports
are longer than 512 tokens; `answerdotai/ModernBERT-base` (which reads longer
texts) is added automatically only above 15%.

The run takes about 15 minutes on the GPU the first time, including about 1 GB
of model downloads. Results go to `results/encoder_comparison/`: one folder per
model plus `summary.md` with the comparison. The report vectors are saved under
`datasets/embeddings/` (not in git) and reused by later runs.

## Dataset adaptation

The three CSVs live in `datasets/processed/A/A_training.csv`,
`A_validation.csv`, and `A_test.csv`. Each holds the input features plus six
`label: <event>` columns (1 = the event was recorded, 0 = not).
Counts, experience, visibility, and hours into shift are continuous inputs:
median imputation and standardization are fitted on training only.
Each remaining checkbox is its own categorical input, using the NBI notebook's
`OrdinalEncoder` and +1 shift (zero is reserved for unknown categories).
No one-hot expansion or temporal sequences are created. Time-of-day checkboxes
are ordinary report attributes, not a history of observations.

The classifier produces six logits, one per event, trained with binary
cross-entropy on each. BERT and text fusion are future work.

Checkpoints go to `checkpoints/<experiment folder>/` unless `--output` names
another folder. It contains one checkpoint (`ft_transformer_seed<seed>.pt`) and
one `history_seed<seed>.csv` per seed, plus `preprocessor.joblib` and
`thresholds.json` (the yes/no cutoff of each event). Each checkpoint includes
model settings, feature order, event names, and the best epoch. Unlike the NBI
notebook, this uses the existing C3RS fixed partitions rather than
bridge-grouped cross-validation.

## Files

- `train.py`: FT-Transformer training, early stopping, checkpoint saving, and
  test scoring, for one seed or the average of several.
- `eval.py`: loss, probabilities, and per-event AP for continuous/categorical
  input batches.
- `utilis/metrics.py`: cutoff selection and test scoring shared by `train.py`
  and the baselines.
- `utilis/results.py`: the results folder of an experiment (never overwritten)
  and its `summary.md`.
- `data/`: existing CSV loading and FT input preprocessing.
- `preprocessing_data.py`: existing C3RS feature, label, and split preparation.
- `scripts/run_baselines.py`: existing baseline comparisons.
- `notebooks/fttransformer.ipynb`: the only notebook on this branch; see
  *Notebook* above.

## Removed from this branch

`models/MLP.py`, the seven EDA notebooks (`notebooks/01_load_data.ipynb`
through `notebooks/07_event_targets.ipynb`), and
`notebooks/C3RS_event_label_review.md` were deleted here. They still exist on
`main`. Merging this branch as it stands would delete them there too, so
restore them on this branch before it is merged into `main`:

```powershell
git checkout main -- models/MLP.py notebooks/0*.ipynb notebooks/C3RS_event_label_review.md
```
