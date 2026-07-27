# Mask credibility audit

Scope: stability and sensitivity only; no vessel ground truth was used.

| Dataset | Perturbation | Mask IoU | Skeleton F1 | Junction repeatability |
|---|---|---:|---:|---:|
| FIMD | photometric | 0.8889 | 0.9461 | 0.8331 |
| FIMD | affine | 0.8475 | 0.9491 | 0.7410 |
| FIRE | photometric | 0.8879 | 0.9449 | 0.8326 |
| FIRE | affine | 0.8226 | 0.9335 | 0.7301 |
| Lab4 | photometric | 0.8419 | 0.9163 | 0.8037 |
| Lab4 | affine | 0.6984 | 0.8487 | 0.6841 |

These values cannot be interpreted as segmentation Dice or anatomical accuracy.
