# baselines: baselines on the structured fields

- LogReg, RandomForest, XGBoost: one yes/no model per event, settings chosen per event by validation AP. Seed 42.
- Run on 2026-11-19.
- All numbers are test scores. Each event's yes/no cutoff was picked on validation.

## Average over the events

| Model | Average F1 | Mean AP |
|---|---|---|
| LogReg | 0.629 | 0.634 |
| RandomForest | 0.615 | 0.649 |
| XGBoost | 0.630 | 0.656 |

## F1 per event

| Event | LogReg | RandomForest | XGBoost |
|---|---|---|---|
| Station Ops | 0.879 | 0.889 | 0.875 |
| Speed - MAS | 0.594 | 0.576 | 0.596 |
| Emerg/Penalty Brake | 0.527 | 0.561 | 0.558 |
| Mechanical (less) | 0.641 | 0.601 | 0.615 |
| Signal - OBCS | 0.606 | 0.587 | 0.610 |
| Speed - TSR | 0.527 | 0.477 | 0.528 |
| Average | 0.629 | 0.615 | 0.630 |

## AP per event

| Event | LogReg | RandomForest | XGBoost |
|---|---|---|---|
| Station Ops | 0.956 | 0.958 | 0.962 |
| Speed - MAS | 0.527 | 0.554 | 0.561 |
| Emerg/Penalty Brake | 0.484 | 0.593 | 0.579 |
| Mechanical (less) | 0.647 | 0.642 | 0.648 |
| Signal - OBCS | 0.662 | 0.631 | 0.648 |
| Speed - TSR | 0.526 | 0.518 | 0.535 |
| Average | 0.634 | 0.649 | 0.656 |

## Precision per event

| Event | LogReg | RandomForest | XGBoost |
|---|---|---|---|
| Station Ops | 0.871 | 0.914 | 0.827 |
| Speed - MAS | 0.473 | 0.510 | 0.479 |
| Emerg/Penalty Brake | 0.455 | 0.514 | 0.583 |
| Mechanical (less) | 0.636 | 0.680 | 0.685 |
| Signal - OBCS | 0.688 | 0.711 | 0.623 |
| Speed - TSR | 0.421 | 0.558 | 0.403 |
| Average | 0.591 | 0.648 | 0.600 |

## Recall per event

| Event | LogReg | RandomForest | XGBoost |
|---|---|---|---|
| Station Ops | 0.887 | 0.866 | 0.929 |
| Speed - MAS | 0.800 | 0.662 | 0.787 |
| Emerg/Penalty Brake | 0.625 | 0.618 | 0.535 |
| Mechanical (less) | 0.647 | 0.538 | 0.558 |
| Signal - OBCS | 0.542 | 0.500 | 0.597 |
| Speed - TSR | 0.703 | 0.416 | 0.766 |
| Average | 0.701 | 0.600 | 0.695 |

## Cutoff picked on validation per event

| Event | LogReg | RandomForest | XGBoost |
|---|---|---|---|
| Station Ops | 0.38 | 0.44 | 0.26 |
| Speed - MAS | 0.28 | 0.34 | 0.27 |
| Emerg/Penalty Brake | 0.27 | 0.28 | 0.35 |
| Mechanical (less) | 0.23 | 0.51 | 0.62 |
| Signal - OBCS | 0.35 | 0.38 | 0.26 |
| Speed - TSR | 0.24 | 0.35 | 0.21 |
