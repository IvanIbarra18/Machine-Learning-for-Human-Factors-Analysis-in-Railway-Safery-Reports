# tabular_only: FT-Transformer on the structured fields

- One model per seed. Seeds [42, 43, 44, 45, 46]; best epoch per seed [14, 19, 23, 17, 22]; loss bce; batch size 64.
- Checkpoints: checkpoints/ft_six_events (not in git). Run on 2026-11-19; the per-seed files were written afterwards from the saved checkpoints and match the run's own scores.
- Single models: average F1 0.635 +/- 0.005, mean AP 0.657 +/- 0.008 (mean +/- SD over 5 seeds).
- All numbers are test scores. Each event's yes/no cutoff was picked on validation.

## Average over the events

| Model | Average F1 | Mean AP |
|---|---|---|
| FTTransformer (5-seed average) | 0.643 | 0.667 |
| Seed 42 | 0.636 | 0.657 |
| Seed 43 | 0.629 | 0.644 |
| Seed 44 | 0.634 | 0.666 |
| Seed 45 | 0.643 | 0.663 |
| Seed 46 | 0.633 | 0.654 |

## F1 per event

| Event | FTTransformer (5-seed average) | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|---|---|
| Station Ops | 0.897 | 0.895 | 0.885 | 0.881 | 0.892 | 0.891 |
| Speed - MAS | 0.605 | 0.590 | 0.596 | 0.596 | 0.603 | 0.601 |
| Emerg/Penalty Brake | 0.551 | 0.542 | 0.542 | 0.531 | 0.535 | 0.541 |
| Mechanical (less) | 0.630 | 0.641 | 0.593 | 0.624 | 0.644 | 0.605 |
| Signal - OBCS | 0.634 | 0.643 | 0.633 | 0.623 | 0.631 | 0.634 |
| Speed - TSR | 0.543 | 0.506 | 0.524 | 0.548 | 0.553 | 0.525 |
| Average | 0.643 | 0.636 | 0.629 | 0.634 | 0.643 | 0.633 |

## AP per event

| Event | FTTransformer (5-seed average) | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|---|---|
| Station Ops | 0.960 | 0.958 | 0.955 | 0.959 | 0.958 | 0.952 |
| Speed - MAS | 0.572 | 0.555 | 0.524 | 0.587 | 0.562 | 0.558 |
| Emerg/Penalty Brake | 0.573 | 0.556 | 0.538 | 0.579 | 0.558 | 0.551 |
| Mechanical (less) | 0.658 | 0.664 | 0.646 | 0.655 | 0.664 | 0.641 |
| Signal - OBCS | 0.669 | 0.664 | 0.664 | 0.652 | 0.672 | 0.649 |
| Speed - TSR | 0.571 | 0.545 | 0.537 | 0.563 | 0.564 | 0.572 |
| Average | 0.667 | 0.657 | 0.644 | 0.666 | 0.663 | 0.654 |

## Precision per event

| Event | FTTransformer (5-seed average) | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|---|---|
| Station Ops | 0.922 | 0.917 | 0.880 | 0.888 | 0.903 | 0.924 |
| Speed - MAS | 0.508 | 0.490 | 0.487 | 0.493 | 0.494 | 0.518 |
| Emerg/Penalty Brake | 0.460 | 0.480 | 0.448 | 0.415 | 0.414 | 0.448 |
| Mechanical (less) | 0.657 | 0.700 | 0.761 | 0.689 | 0.686 | 0.592 |
| Signal - OBCS | 0.616 | 0.662 | 0.652 | 0.652 | 0.566 | 0.593 |
| Speed - TSR | 0.463 | 0.407 | 0.441 | 0.451 | 0.443 | 0.538 |
| Average | 0.604 | 0.609 | 0.611 | 0.598 | 0.584 | 0.602 |

## Recall per event

| Event | FTTransformer (5-seed average) | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|---|---|
| Station Ops | 0.874 | 0.874 | 0.889 | 0.874 | 0.882 | 0.861 |
| Speed - MAS | 0.748 | 0.741 | 0.770 | 0.754 | 0.774 | 0.715 |
| Emerg/Penalty Brake | 0.687 | 0.622 | 0.687 | 0.738 | 0.756 | 0.684 |
| Mechanical (less) | 0.606 | 0.590 | 0.486 | 0.570 | 0.606 | 0.618 |
| Signal - OBCS | 0.653 | 0.625 | 0.616 | 0.597 | 0.713 | 0.681 |
| Speed - TSR | 0.656 | 0.670 | 0.646 | 0.699 | 0.737 | 0.512 |
| Average | 0.704 | 0.687 | 0.682 | 0.705 | 0.745 | 0.678 |

## Cutoff picked on validation per event

| Event | FTTransformer (5-seed average) | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|---|---|
| Station Ops | 0.52 | 0.63 | 0.31 | 0.44 | 0.46 | 0.44 |
| Speed - MAS | 0.32 | 0.31 | 0.35 | 0.27 | 0.33 | 0.28 |
| Emerg/Penalty Brake | 0.27 | 0.28 | 0.31 | 0.18 | 0.26 | 0.24 |
| Mechanical (less) | 0.29 | 0.34 | 0.47 | 0.35 | 0.31 | 0.29 |
| Signal - OBCS | 0.28 | 0.36 | 0.31 | 0.37 | 0.19 | 0.32 |
| Speed - TSR | 0.27 | 0.18 | 0.26 | 0.30 | 0.19 | 0.43 |
