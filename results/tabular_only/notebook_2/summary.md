# tabular_only/notebook_2: FT-Transformer on the structured fields

- FTTransformer with default settings, structured fields only, leakage fields removed.
- Seed 42, batch 32, best epoch 11 (validation mean AP 0.607).
- Micro F1 0.641, macro F1 0.629.
- All numbers are test scores. Each event's yes/no cutoff was picked on validation.

## Average over the events

| Model | Average F1 | Mean AP |
|---|---|---|
| FTTransformer | 0.629 | 0.647 |

## F1 per event

| Event | FTTransformer |
|---|---|
| Station Ops | 0.895 |
| Speed - MAS | 0.599 |
| Emerg/Penalty Brake | 0.546 |
| Mechanical (less) | 0.619 |
| Signal - OBCS | 0.596 |
| Speed - TSR | 0.520 |
| Average | 0.629 |

## AP per event

| Event | FTTransformer |
|---|---|
| Station Ops | 0.957 |
| Speed - MAS | 0.545 |
| Emerg/Penalty Brake | 0.564 |
| Mechanical (less) | 0.634 |
| Signal - OBCS | 0.638 |
| Speed - TSR | 0.544 |
| Average | 0.647 |

## Precision per event

| Event | FTTransformer |
|---|---|
| Station Ops | 0.906 |
| Speed - MAS | 0.499 |
| Emerg/Penalty Brake | 0.489 |
| Mechanical (less) | 0.701 |
| Signal - OBCS | 0.573 |
| Speed - TSR | 0.409 |
| Average | 0.596 |

## Recall per event

| Event | FTTransformer |
|---|---|
| Station Ops | 0.884 |
| Speed - MAS | 0.751 |
| Emerg/Penalty Brake | 0.618 |
| Mechanical (less) | 0.554 |
| Signal - OBCS | 0.620 |
| Speed - TSR | 0.713 |
| Average | 0.690 |

## Cutoff picked on validation per event

| Event | FTTransformer |
|---|---|
| Station Ops | 0.48 |
| Speed - MAS | 0.29 |
| Emerg/Penalty Brake | 0.33 |
| Mechanical (less) | 0.43 |
| Signal - OBCS | 0.29 |
| Speed - TSR | 0.22 |
