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

## CNN Architecture and Design Choices

The network is a fully convolutional CNN that takes a 32x32 grayscale shuffled image and outputs a 16x4x4 probability map, where each of the 16 channels represents the likelihood that a given patch originally belonged to that position. The architecture consists of four convolutional blocks. Each block contains two 3x3 convolutional layers with ReLU activations, followed by 2x2 max pooling that halves the spatial dimensions (32x32 → 16x16 → 8x8 → 4x4). Channel depth increases through the network (32 → 64 → 128) before narrowing back to 64, then a 1x1 convolution reduces to 16 channels matching the number of patch positions. A softmax layer normalises the output across channels so each spatial location receives a probability distribution over the 16 possible original positions.

The key design choice is the addition of channel attention modules after each max-pooling layer. Each module performs global average pooling to compress the spatial dimensions into a single value per channel, passes this through two fully connected layers (with a reduction factor of 4) using ReLU and sigmoid activations, and multiplies the result back into each channel. This allows the network to learn which feature channels are most informative at each spatial resolution — for example, edge detectors may matter more at fine scales while broader structural features matter at coarser scales. A final patch assignment step uses the Hungarian algorithm to find the optimal one-to-one mapping between shuffled positions and original positions, ensuring each patch is assigned exactly once.

The fully convolutional design was chosen over a fully connected head because the output naturally maps to a 4x4 spatial grid matching the patch layout, preserving spatial correspondence and avoiding a large parameter count.

## Training Procedure

The model was trained using cross-entropy loss over 16-class patch position predictions, excluding all-black patches (pixels below 0.05) from accuracy since identical empty patches cannot be distinguished.

Training began with SGD (learning rate 1e-3, batch size 4) on the baseline CNN without channel attention, reaching ~49% validation accuracy. Switching to Adam (learning rate 1e-3, batch size 16) with reduce-on-plateau scheduling (halve LR after 10 stagnant epochs, minimum 1e-6) improved this to 59.6% over ~300 epochs with progressively lower learning rates. To break through this plateau, two architectural changes were tested: widening the bottleneck from 128 to 256 channels, and adding channel attention modules. The wider network reached only 52.7% and was significantly slower to train, while channel attention — with the nine pre-trained convolutional weights transferred into the new architecture — reached 68.1% in 68 further epochs.

Data augmentation was applied before patch shuffling: random horizontal flips, rotation (±15°, black border fill), foreground-only brightness/contrast shifts (±30%), and Gaussian blur. To improve generalisation to unseen image distributions, morphological erosion and dilation were added to simulate varying stroke thicknesses — the training data (MNIST-style digits) has relatively thick, uniform strokes, while other handwritten digit datasets like USPS and Kaggle Archive have thinner, dimmer strokes. By randomly eroding and dilating during training, the model learns to handle a wider range of stroke styles without needing additional training data. All augmentations are applied only to foreground pixels and backgrounds are preserved via a final threshold cleanup (pixels below 0.05 set to zero). Experiments with inverted dropout (p=0.15) did not improve generalisation and were not used in the final model.

## Evaluation

Accuracy is measured per image as the fraction of non-black patches assigned to their correct original position, using the Hungarian algorithm for optimal one-to-one assignment. All-black patches (mean pixel intensity below 0.05) are excluded since identical empty patches carry no positional information.

| Phase | Epochs | Built On | Val Accuracy |
|---|---|---|---|
| SGD baseline | 120 | — | 49% |
| Adam + LR scheduling | ~300 | SGD checkpoint | 59.6% |
| Wider bottleneck (256ch) | 22 | From scratch | 52.7% |
| Channel attention (SE) | 68 | Adam checkpoint | **68.1%** |

Each phase addressed a specific bottleneck: SGD was too slow to converge, Adam with learning rate decay pushed further but plateaued at the architecture's limit, and channel attention broke through by allowing the network to weight informative features at each spatial scale. The wider bottleneck was trained independently but underperformed and was abandoned. Figure 1 shows example reconstructions from the final model — correctly reconstructed images tend to have distinctive patches with clear stroke features, while errors occur when patches are visually similar or mostly empty.

## Reflections and Improvements

The final model was evaluated on three unseen datasets to test generalisation:

| Dataset | Resolution | Accuracy |
|---|---|---|
| Validation (in-domain) | 32x32 | **68.1%** |
| QMNIST | 28x28 → 32x32 | **68.9%** |
| Kaggle Archive | 28x28 → 32x32 | 27.0% |
| USPS | 16x16 → 32x32 | 26.2% |

QMNIST matches in-domain accuracy, confirming generalisation to similar distributions. Archive and USPS drop significantly — as shown in Figure 2, Archive digits have thinner strokes and lower pixel intensity, while USPS images require 4x upscaling from 16x16 which blurs fine spatial details the model relies on for patch discrimination. Fine-tuning with morphological augmentation (erosion/dilation) improved USPS to 31.8% but reduced validation accuracy to 64.1%, so the original model was kept. The main improvement would be to train on diverse handwritten digit sources, exposing the model to varied stroke styles and resolutions rather than relying on augmentation to simulate this variation.

Training efficiency was also a challenge. The initial im2col used triple-nested Python loops; replacing it with NumPy stride tricks (as_strided) and rewriting col2im with slice-based accumulation eliminated this bottleneck. Cross-entropy loss was optimised by replacing one-hot encoding with direct gather indexing, reducing memory from O(B·H·W·C) to O(B·H·W).
