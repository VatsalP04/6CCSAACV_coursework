# Training Journey & Experiment Log

## Starting Point

We started with a from-scratch CNN implementation in NumPy for a patch-shuffling task: 32x32 grayscale images get split into a 4x4 grid (16 patches of 8x8), randomly shuffled, and the model has to predict where each patch originally came from.

The initial codebase was a single Jupyter notebook that got split into focused modules: `layers.py` (all layer implementations), `train.py` (training loop), `utils.py` (data loading, metrics), and `test.py` (submission script).

---

## Phase 1: SGD Training — Hit a Ceiling (~55%)

The first real training used SGD with a small dataset (2k images, 2x2 patches as warm-up, then 4x4). Over 120 epochs we got to about **49% val accuracy** and it plateaued hard. Switching to the full 10k dataset and adding basic augmentation (rotation, brightness, contrast, blur) pushed it to **~55%**, but then it hit another wall.

The training logs showed the loss barely moving epoch to epoch — classic sign of the optimizer struggling.

## Phase 2: Switching to Adam + LR Scheduling (~59%)

We identified three problems:
1. **SGD was too slow** — Adam would converge faster
2. **Batch size of 4 was too small** — noisy gradients, increased to 16
3. **No learning rate scheduling** — added reduce-on-plateau (halve LR when val loss stalls for N epochs)

We refactored `train.py` to support chained runs:
- **Run 1**: Adam lr=1e-3, batch=16, 50 epochs with default augmentation
- **Run 2**: Adam lr=5e-4, batch=32, 40 epochs with stronger augmentation (more rotation, brightness, contrast, blur)

This got us to **~59% val accuracy**. We extended Run 2 for another 100 epochs and watched 5 LR decays (from 2.5e-4 down to 8e-6), eventually reaching **59.6%**. But it was clearly plateauing — the architecture was the bottleneck now.

## Phase 3: Architecture Experiments

With the optimizer and training pipeline maxed out, we needed architectural changes. We ran two experiments simultaneously:

### run_wider.py — Wider Bottleneck (256 channels)
Doubled the bottleneck from 128 to 256 channels. Trained from scratch. Reached 52.7% in 22 epochs but was very slow (~11 min/epoch) and didn't seem like it would beat the baseline by much.

### run_global_context.py — Squeeze-and-Excite (SE Blocks)
Added `GlobalContextLayer` after each MaxPool — this is a squeeze-and-excite mechanism: global average pooling -> FC -> ReLU -> FC -> sigmoid -> channel-wise scaling. It lets the network learn which channels matter at each spatial resolution.

Key decision: we **transferred conv weights** from the best existing checkpoint (9 conv layers matched perfectly) into the new architecture. This meant the SE blocks started learning on top of already-trained features.

Results were immediately better:
- **Epoch 1: 60.0%** (already surpassed the old best of 59.6%)
- **Epoch 10: 64.4%**
- **Epoch 38: 68.1%** and still climbing

The global context model was clearly the winner — the SE blocks gave the network the ability to attend to relevant channels at each scale, which is exactly what you need for patch position prediction.

## Phase 4: Testing & Evaluation

With the global context model as our best (`final.pkl`), we started evaluating:

### Validation Set (Training Distribution)
- Used `experiments/eval_model.py` which shuffles val images, runs `test.py`, and compares predictions to ground truth
- **~60% accuracy** on the validation split (this was from the earlier checkpoint; the GC model would be higher)

### Unseen Dataset: QMNIST
- 28x28 handwritten digits, resized to 32x32
- **67% accuracy** on 1000 images — great generalisation since QMNIST is closest to our training data (both MNIST-style)

### Unseen Dataset: USPS
- Only 16x16 images, had to upscale to 32x32
- Tested 4 interpolation methods: NEAREST (13%), BILINEAR (25%), CUBIC (23.5%), LANCZOS4 (25.4%)
- **~25% accuracy** — fundamental distribution mismatch from aggressive upscaling, not much we could do

### Unseen Dataset: Archive (Kaggle MNIST)
- 107k images, 28x28 PNGs
- Initially got **0% accuracy** — turned out the digit data was stored in the **alpha channel** (RGBA format), not RGB. OpenCV was reading all-black images.
- After fixing the loader (`cv.IMREAD_UNCHANGED` + extract channel 3): **26.3% accuracy**
- Still low compared to QMNIST — archive digits have thinner strokes and different pixel distribution

## Phase 5: Generalisation Problem

We compared the pixel statistics:

| Dataset | Mean Intensity | Std | Non-zero Pixels |
|---|---|---|---|
| Training | 0.137 | 0.299 | 20.7% |
| Archive | 0.085 | 0.234 | 14.8% |

Archive digits are thinner (less non-zero pixels) and dimmer (lower mean). The model had never seen this kind of variation.

### First Attempt: Test-Time Preprocessing
Tried histogram equalisation, CLAHE, per-image normalisation on archive images before inference. Barely helped (~29.7% best vs 28.1% raw).

### Second Attempt: Training-Time Augmentation
Added two new augmentations to `augment_image()`:
1. **Morphological erosion/dilation** (morph_prob=0.5) — randomly erode (thinner strokes) or dilate (thicker strokes) with 2x2 or 3x3 kernels
2. **Gaussian noise** (noise_prob=0.3) — adds pixel-level noise to foreground only

### The Background Problem
First version of augmentation was bad — brightness shifts and noise were applied to the entire image, turning black backgrounds grey. The augmented data had 62% non-zero pixels vs 20% original. Completely wrong direction.

Fixed this by making brightness, contrast, and noise **foreground-only** (only affect pixels > 0.05). Also changed rotation from `BORDER_REFLECT_101` (which reflects digit pixels into the background) to `BORDER_CONSTANT` with value 0 (clean black fill). Added a final cleanup threshold: any pixel < 13 (0.05 * 255) gets set to 0.

This matters because the coursework scoring **excludes all-black patches** from accuracy. If augmentation corrupts the background, those patches become non-black and the model gets penalised for guessing their positions.

After fixes:

| Dataset | Mean | Std | Non-zero |
|---|---|---|---|
| Training original | 0.141 | 0.302 | 21.0% |
| Training augmented | 0.142 | 0.260 | 24.6% |

Clean backgrounds preserved, but the eroded samples now produce thinner strokes similar to archive.

## Phase 6: Generalisation Training — Two Parallel Approaches

With the augmentation pipeline fixed, we launched two experiments simultaneously to find the best strategy for unseen data:

### Approach A: GC + Generalisation Augmentation (`run_generalise.py`)
- Resumed from the best GC checkpoint (epoch 68)
- Fine-tuned with Adam lr=5e-4, 80 epochs, batch=16
- Full augmentation suite: morph_prob=0.5, noise_prob=0.3, brightness=0.4, contrast=0.4, rotation=15, blur=0.4
- All foreground-only, backgrounds stay black
- Idea: teach the existing architecture to handle stroke thickness variation through data augmentation alone

### Approach B: GC + Dropout (`run_dropout.py`)
- New architecture: added `DropoutLayer(p=0.15)` after the first ReLU in each conv block (4 dropout layers total)
- Implemented inverted dropout in `layers.py`: scales by 1/(1-p) during training, identity at test time
- Added `train_mode()` / `eval_mode()` to Network class to toggle dropout
- Transferred both conv weights (9 layers) and GC weights (3 layers) from the best GC checkpoint
- Adam lr=5e-4, 80 epochs, batch=16, same augmentation as Approach A
- Idea: dropout forces the network to not rely on specific feature combinations, improving generalisation

### Bug Fix: eval_mode() Missing
Discovered that `eval_unseen.py` and `visualise.py` were not calling `net.eval_mode()` after loading models. This meant dropout layers stayed active during evaluation, randomly zeroing 15% of activations. Fixed by adding `if hasattr(net, 'eval_mode'): net.eval_mode()` after pickle load in both files + `test.py`.

## Phase 7: Final Evaluation — Comprehensive Model Comparison

Evaluated all three approaches across all four datasets (1000 images each):

### Global Context Only (epoch 68) — Best Overall

| Dataset | Accuracy |
|---|---|
| **Original (Val)** | **68.09%** |
| **QMNIST** | **68.88%** |
| Archive | 27.00% |
| USPS | 26.19% |

### GC + Augmentation (epoch 32) — Best USPS

| Dataset | Accuracy |
|---|---|
| Original (Val) | 64.14% |
| QMNIST | 67.69% |
| Archive | 26.53% |
| **USPS** | **31.78%** |

### GC + Dropout (epoch 25) — Weakest

| Dataset | Accuracy |
|---|---|
| Original (Val) | 59.67% |
| QMNIST | 62.79% |
| Archive | 25.60% |
| USPS | 29.46% |

### Analysis

1. **SE blocks are doing the heavy lifting.** The original GC model with no extra regularisation remains the strongest overall — the squeeze-and-excite mechanism gives the network exactly the channel attention it needs for patch position prediction.

2. **Dropout hurt performance.** Rather than improving generalisation, dropout (even at a light p=0.15) reduced accuracy across all datasets. The model hadn't converged enough in 25 epochs to overcome the regularisation penalty, and the task may not benefit from dropout — patch position prediction relies on precise spatial features that dropout disrupts.

3. **Augmentation helped USPS but traded off in-domain accuracy.** The morphological erosion/dilation augmentation did its job — USPS jumped from 26% to 32% because the model learned to handle different stroke thicknesses. But it came at the cost of ~4% on the validation set and ~1% on QMNIST, suggesting the augmentation added noise to the training signal.

4. **Archive remains stuck at ~26-27%.** None of the approaches significantly moved archive accuracy. The distribution gap (thinner strokes, dimmer intensity, alpha channel storage) is too large for augmentation alone to bridge without explicitly training on similar data.

### Decision: Best Model for Submission

**Global Context (epoch 68)** — `epoch_global_context_epoch_68.pkl`

- Highest validation accuracy (68.09%) — well above the 60% threshold for full marks
- Highest QMNIST accuracy (68.88%) — best generalisation to the closest unseen distribution
- Archive and USPS are low across all models anyway, so the trade-off from augmentation (lose 4% val for 5% USPS) isn't worth it

---

## Key Milestones

| Phase | Best Val Acc | What Changed |
|---|---|---|
| SGD, 2k data | 49% | Baseline |
| SGD, 10k + aug | 55% | More data, augmentation |
| Adam + LR decay | 59.6% | Optimizer, scheduling |
| Global Context (SE) | 68.1% | Architecture (SE blocks + weight transfer) |
| GC + Augmentation | 64.1% | Morph/noise augmentation (better USPS, worse val) |
| GC + Dropout | 59.7% | Dropout regularisation (hurt all metrics) |

### Grading Thresholds (from coursework)
- <30% = 0 marks
- 30-40% = 35 marks
- 40-50% = 42 marks
- 50-60% = 50 marks
- **>=60% = 70 marks (full marks)**

The global context model at 68.09% val accuracy exceeds the 60% threshold for full marks.

---

## Training Procedure Summary

Training progressed through a series of experiments, each building on the last. `run_base.py` trained the baseline CNN with SGD (lr=1e-3, batch=4, no augmentation), reaching ~49%. `run1_adam.py` switched to Adam (lr=1e-3, batch=16) with basic augmentation, pushing to ~55%. `run2_strong_aug_continue_from_run1.py` fine-tuned from Run 1's best checkpoint with lower LR (5e-4), larger batch (32), and stronger augmentation (rotation ±15°, brightness/contrast ±30%). `run2_continue.py` extended this for another 100 epochs with further LR decay, plateauing at ~59.6%.

The best model came from `run_global_context.py`, which added squeeze-and-excite (SE) blocks after each pooling layer. It transferred the 9 pre-trained conv layer weights from Run 2's best checkpoint into the new architecture, so the SE blocks learned on top of already-trained features. This reached **68.1% val accuracy** in 68 epochs.

Two further experiments attempted to improve generalisation: `run_generalise.py` (morphological augmentation) and `run_dropout.py` (dropout regularisation). Neither surpassed the original GC model on the validation set. All experiment scripts import the shared training loop from `train.py` and data utilities from `utils.py`.
