# fusion_frozen: frozen FT-Transformer and BERT, two-tower join

- Both encoders frozen. FT-Transformer CLS vector and BERT CLS vector, each projected to 256, normalized, concatenated, then dropout and one linear layer with six outputs.
- Seed 42, batch 64, best epoch 8 (validation mean AP 0.866).
- Encoders alone on validation: FT-Transformer mean AP 0.607, BERT mean AP 0.868.
- Micro F1 0.845, macro F1 0.836.
- All numbers are test scores. Each event's yes/no cutoff was picked on validation.

## Average over the events

| Model | Average F1 | Mean AP |
|---|---|---|
| Fusion (frozen) | 0.836 | 0.894 |

## F1 per event

| Event | Fusion (frozen) |
|---|---|
| Station Ops | 0.940 |
| Speed - MAS | 0.810 |
| Emerg/Penalty Brake | 0.888 |
| Mechanical (less) | 0.738 |
| Signal - OBCS | 0.767 |
| Speed - TSR | 0.873 |
| Average | 0.836 |

## AP per event

| Event | Fusion (frozen) |
|---|---|
| Station Ops | 0.983 |
| Speed - MAS | 0.852 |
| Emerg/Penalty Brake | 0.937 |
| Mechanical (less) | 0.813 |
| Signal - OBCS | 0.858 |
| Speed - TSR | 0.917 |
| Average | 0.894 |

## Precision per event

| Event | Fusion (frozen) |
|---|---|
| Station Ops | 0.930 |
| Speed - MAS | 0.817 |
| Emerg/Penalty Brake | 0.879 |
| Mechanical (less) | 0.725 |
| Signal - OBCS | 0.780 |
| Speed - TSR | 0.831 |
| Average | 0.827 |

## Recall per event

| Event | Fusion (frozen) |
|---|---|
| Station Ops | 0.950 |
| Speed - MAS | 0.803 |
| Emerg/Penalty Brake | 0.898 |
| Mechanical (less) | 0.751 |
| Signal - OBCS | 0.755 |
| Speed - TSR | 0.919 |
| Average | 0.846 |

## Cutoff picked on validation per event

| Event | Fusion (frozen) |
|---|---|
| Station Ops | 0.35 |
| Speed - MAS | 0.54 |
| Emerg/Penalty Brake | 0.38 |
| Mechanical (less) | 0.35 |
| Signal - OBCS | 0.43 |
| Speed - TSR | 0.21 |
