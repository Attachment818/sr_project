# Anatomy-Context Feature Learning: research specification v1

## Problem hypothesis

SuperRetina/G0 learns a local descriptor mainly from detector-selected points
under synthetic geometry. FIMD failures indicate that the local pixels at a
true anatomical correspondence may no longer be trustworthy after pathology
or acquisition changes. The new hypothesis is that point identity must be
estimated jointly from local detail and long-range retinal context.

## Evidence from related mechanisms

- R2D2 separates repeatability from descriptor reliability; salience alone is
  not matchability: https://arxiv.org/abs/1906.06195
- DeDoDe argues that detector learning should not be defined by the current
  descriptor's nearest-neighbour proxy: https://arxiv.org/abs/2308.08479
- RoMa combines robust coarse foundation features with fine ConvNet features,
  showing the tension between identity and localization:
  https://arxiv.org/abs/2305.15404
- RetinaRegNet and DINO-Reg show that diffusion/foundation representations can
  provide cross-appearance medical correspondence:
  https://arxiv.org/abs/2404.16017 and https://arxiv.org/abs/2402.15687
- MIFNet learns modality-invariant matching features without aligned target
  modality training pairs: https://arxiv.org/abs/2501.11299
- DenseCL demonstrates that self-supervised objectives can target spatial
  features rather than only a global image vector:
  https://arxiv.org/abs/2011.09157

These works are mechanism references, not an implementation template.

## Proposed representation

The extractor produces at 1/8 resolution:

1. a local feature for accurate localization;
2. a context feature aggregated from 1/16 and 1/32 receptive fields;
3. a spatial local/context gate;
4. a normalized fused descriptor;
5. separate repeatability and reliability predictions.

The current prototype is isolated under `model/anatomy_context/` and does not
call PKE, vessel-mask generation, value maps, or SuperRetina training code.

## Training phases

### Phase A - masked-evidence anatomy pretraining

An EMA teacher observes the clean image. The student observes the same image
with geometry-preserving acquisition changes and irregular missing-evidence
regions. The student predicts the clean teacher's feature field, particularly
where local evidence is unavailable. The masks are explicitly treated as
missing evidence, not as simulated pathology.

Feature variance and covariance regularization are required to prevent a
collapsed constant representation.

### Phase B - representation factorization

Separate stable anatomy information from appearance information. Appearance
changes should preserve the anatomy component while an auxiliary reconstruction
or teacher-prediction target prevents loss of fine information.

### Phase C - decoupled point selection

Train repeatability under known geometry and reliability from correspondence
ambiguity. The final score must distinguish a repeatable point from a uniquely
matchable point. No morph vessel mask is used as ground truth.

### Phase D - short geometry calibration

Only after anatomy pretraining, apply homography/affine/TPS pairs to calibrate
equivariance and subpixel localization. Synthetic geometry is a final
calibration task, not the source of the representation from scratch.

## Required evidence before full training

1. Local, context, gate, and descriptor branches receive finite non-zero
   gradients in a CPU smoke test.
2. The EMA teacher is frozen and updated only through EMA.
3. The representation does not collapse on a small real-image pilot.
4. Frozen descriptors improve positive rank or bidirectional consistency on
   selected FIRE/FIMD pairs before a detector is trained.
5. Any gain must be separated into identity robustness, localization accuracy,
   and matchability calibration.

## Initial ablations

- local-only versus context-only versus gated fusion;
- clean-view consistency versus masked-evidence prediction;
- one context scale versus two context scales;
- repeatability-only versus repeatability times reliability;
- Lab4-only versus additional unlabeled retinal data.

No vessel-weight, PKE-threshold, or G7 gate scan belongs to this research line.
