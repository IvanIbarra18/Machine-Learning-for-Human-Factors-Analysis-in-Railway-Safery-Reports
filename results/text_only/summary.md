# text_only: BERT on the reporter narratives

- bert-base-uncased, CLS vector, dropout, one linear layer with six outputs.
- Reporter narratives only, cut to 512 tokens.
- Seed 42, batch 16 x 1 accumulation, best epoch 5 (validation mean AP 0.868).
- Micro F1 0.843, macro F1 0.836.
- All numbers are test scores. Each event's yes/no cutoff was picked on validation.

## Average over the events

| Model | Average F1 | Mean AP |
|---|---|---|
| BERT | 0.836 | 0.892 |

## F1 per event

| Event | BERT |
|---|---|
| Station Ops | 0.941 |
| Speed - MAS | 0.817 |
| Emerg/Penalty Brake | 0.884 |
| Mechanical (less) | 0.712 |
| Signal - OBCS | 0.785 |
| Speed - TSR | 0.877 |
| Average | 0.836 |

## AP per event

| Event | BERT |
|---|---|
| Station Ops | 0.982 |
| Speed - MAS | 0.872 |
| Emerg/Penalty Brake | 0.934 |
| Mechanical (less) | 0.782 |
| Signal - OBCS | 0.851 |
| Speed - TSR | 0.930 |
| Average | 0.892 |

## Precision per event

| Event | BERT |
|---|---|
| Station Ops | 0.957 |
| Speed - MAS | 0.799 |
| Emerg/Penalty Brake | 0.881 |
| Mechanical (less) | 0.670 |
| Signal - OBCS | 0.761 |
| Speed - TSR | 0.865 |
| Average | 0.822 |

## Recall per event

| Event | BERT |
|---|---|
| Station Ops | 0.926 |
| Speed - MAS | 0.836 |
| Emerg/Penalty Brake | 0.887 |
| Mechanical (less) | 0.759 |
| Signal - OBCS | 0.810 |
| Speed - TSR | 0.890 |
| Average | 0.851 |

## Cutoff picked on validation per event

| Event | BERT |
|---|---|
| Station Ops | 0.82 |
| Speed - MAS | 0.53 |
| Emerg/Penalty Brake | 0.40 |
| Mechanical (less) | 0.36 |
| Signal - OBCS | 0.44 |
| Speed - TSR | 0.37 |
