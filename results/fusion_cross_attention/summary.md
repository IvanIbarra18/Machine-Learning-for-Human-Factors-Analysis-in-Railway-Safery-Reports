# fusion_cross_attention: frozen FT-Transformer and BERT, cross-attention then two-tower join

- Both encoders frozen. BERT CLS vector attends to the FT-Transformer field tokens (8 heads), is added back and normalized. That vector and the FT-Transformer CLS vector are each projected to 256, normalized, concatenated, then dropout and one linear layer with six outputs.
- Seed 42, batch 32, best epoch 4 (validation mean AP 0.864).
- Encoders alone on validation: FT-Transformer mean AP 0.607, BERT mean AP 0.868.
- Micro F1 0.845, macro F1 0.837.
- All numbers are test scores. Each event's yes/no cutoff was picked on validation.

## Average over the events

| Model | Average F1 | Mean AP |
|---|---|---|
| Cross-attention (frozen) | 0.837 | 0.890 |

## F1 per event

| Event | Cross-attention (frozen) |
|---|---|
| Station Ops | 0.940 |
| Speed - MAS | 0.818 |
| Emerg/Penalty Brake | 0.877 |
| Mechanical (less) | 0.739 |
| Signal - OBCS | 0.779 |
| Speed - TSR | 0.870 |
| Average | 0.837 |

## AP per event

| Event | Cross-attention (frozen) |
|---|---|
| Station Ops | 0.981 |
| Speed - MAS | 0.855 |
| Emerg/Penalty Brake | 0.935 |
| Mechanical (less) | 0.803 |
| Signal - OBCS | 0.854 |
| Speed - TSR | 0.914 |
| Average | 0.890 |

## Precision per event

| Event | Cross-attention (frozen) |
|---|---|
| Station Ops | 0.930 |
| Speed - MAS | 0.796 |
| Emerg/Penalty Brake | 0.888 |
| Mechanical (less) | 0.707 |
| Signal - OBCS | 0.784 |
| Speed - TSR | 0.890 |
| Average | 0.833 |

## Recall per event

| Event | Cross-attention (frozen) |
|---|---|
| Station Ops | 0.950 |
| Speed - MAS | 0.843 |
| Emerg/Penalty Brake | 0.865 |
| Mechanical (less) | 0.775 |
| Signal - OBCS | 0.773 |
| Speed - TSR | 0.852 |
| Average | 0.843 |

## Cutoff picked on validation per event

| Event | Cross-attention (frozen) |
|---|---|
| Station Ops | 0.26 |
| Speed - MAS | 0.28 |
| Emerg/Penalty Brake | 0.40 |
| Mechanical (less) | 0.28 |
| Signal - OBCS | 0.35 |
| Speed - TSR | 0.63 |
