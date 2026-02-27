import os
import pickle
import random
import numpy as np
import cv2 as cv
from datetime import datetime
from tqdm import tqdm
from layers import (
    Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer, CrossEntropyLoss, Adam
)
from utils import (
    PatchShuffleDataLoader, compute_total_receptive_field,
    setup_logging, assign_patches, compute_reconstruction_accuracy
)


# ---------------------------------------------------------------------------
# Data augmentation (grayscale HWC uint8, applied BEFORE patch shuffle)
# ---------------------------------------------------------------------------
def augment_image(image, p_flip=0.5, max_rotation=10,
                  brightness_range=0.2, contrast_range=0.2, blur_prob=0.3,
                  morph_prob=0.4, noise_prob=0.3):
    """Apply random augmentations to a grayscale image (H, W, C) uint8.

    All augmentations preserve the black background: rotation uses black
    border fill, and a final threshold pass ensures low-value pixels from
    interpolation artefacts are cleaned to pure black.
    """
    h, w = image.shape[:2]

    if random.random() < p_flip:
        image = np.ascontiguousarray(image[:, ::-1, :])

    angle = random.uniform(-max_rotation, max_rotation)
    if abs(angle) > 0.5:
        M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv.warpAffine(image, M, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=0)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

    # Morphological augmentation: random erosion or dilation to vary stroke thickness
    if random.random() < morph_prob:
        ksize = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        img_2d = image[:, :, 0] if image.ndim == 3 else image
        if random.random() < 0.5:
            img_2d = cv.erode(img_2d, kernel, iterations=1)
        else:
            img_2d = cv.dilate(img_2d, kernel, iterations=1)
        image = img_2d[:, :, np.newaxis] if image.ndim == 3 else img_2d

    # Foreground-only brightness
    img_f = image.astype(np.float32)
    fg_mask = img_f > 12.0  # current foreground
    shift = random.uniform(-brightness_range, brightness_range) * 255
    img_f[fg_mask] += shift
    image = np.clip(img_f, 0, 255).astype(np.uint8)

    # Foreground-only contrast
    img_f = image.astype(np.float32)
    factor = random.uniform(1.0 - contrast_range, 1.0 + contrast_range)
    if fg_mask.any():
        fg_mean = img_f[fg_mask].mean()
        img_f[fg_mask] = (img_f[fg_mask] - fg_mean) * factor + fg_mean
    image = np.clip(img_f, 0, 255).astype(np.uint8)

    if random.random() < blur_prob:
        ksize = random.choice([3, 5])
        image = cv.GaussianBlur(image, (ksize, ksize), 0)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

    # Foreground-only Gaussian noise
    if random.random() < noise_prob:
        img_f = image.astype(np.float32)
        fg_now = img_f > 12.0
        sigma = random.uniform(5, 15)
        noise = np.random.randn(*image.shape) * sigma
        img_f[fg_now] += noise[fg_now]
        image = np.clip(img_f, 0, 255).astype(np.uint8)

    # Final cleanup: threshold low-value pixels to pure black.
    # This removes interpolation artefacts from rotation/blur that would
    # otherwise make black patches non-black (penalised in evaluation).
    image[image < 13] = 0  # 13/255 ≈ 0.05

    return image


def augmented_train_batches(loader, aug_params=None):
    """Yield augmented training batches. Augmentation runs before patch shuffle."""
    if aug_params is None:
        aug_params = {}

    files = list(loader.train_files)
    if loader.shuffle:
        random.shuffle(files)

    for start in range(0, len(files), loader.batch_size):
        batch_files = files[start:start + loader.batch_size]
        X, Y = [], []
        for fname in batch_files:
            full_path = os.path.join(loader.dataset_dir, fname)
            image = cv.imread(full_path)
            if image is None:
                continue
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = cv.resize(image, loader.image_size)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]

            image = augment_image(image, **aug_params)

            image, labels = loader._shuffle_patches(image)
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))

            X.append(image)
            Y.append(labels)

        if len(X) > 0:
            yield np.stack(X, axis=0), np.stack(Y, axis=0)


# ---------------------------------------------------------------------------
# Network builder
# ---------------------------------------------------------------------------
def build_network(lr=1e-3):
    """Construct and return the patch-sorting CNN (Week 5 architecture)."""
    return Network([
        ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),   # 32x32 -> 16x16

        ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),   # 16x16 -> 8x8

        ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),   # 8x8 -> 4x4

        ConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),

        ConvLayer(in_channels=64, out_channels=16, kernel_size=1, stride=1, pad=0),

        SoftmaxLayer(axis=1),  # softmax across channel/class dimension
    ], optimizer=Adam(lr=lr))


# ---------------------------------------------------------------------------
# Training loop (reusable across runs)
# ---------------------------------------------------------------------------
def train_run(net, loader, start_epoch, end_epoch, run_dir, run_name,
              use_augmentation, num_patches, lr_patience, lr_factor, lr_min,
              log, aug_params=None):
    """
    Run training for [start_epoch, end_epoch) and return the path to the best checkpoint.
    """
    os.makedirs(run_dir, exist_ok=True)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_path = None

    for epoch in range(start_epoch, end_epoch):
        # ---- Train ----
        net.train_mode()
        epoch_loss = 0.0
        iter_train = 0

        if use_augmentation:
            train_iter = augmented_train_batches(loader, aug_params=aug_params)
        else:
            train_iter = loader.train_batches()

        pbar = tqdm(train_iter, desc=f"Epoch {epoch+1}/{end_epoch} [train]", leave=False)
        for X_batch, Y_batch in pbar:
            iter_train += 1
            out_batch = net.forward(X_batch)

            loss_layer = CrossEntropyLoss()
            loss = loss_layer.forward(out_batch, Y_batch)
            epoch_loss += loss
            pbar.set_postfix(loss=f"{loss:.4f}")

            dX = loss_layer.backward()
            net.backward(dX)
            net.step()

        avg_train_loss = epoch_loss / max(iter_train, 1)

        # ---- Validation (no augmentation) ----
        net.eval_mode()
        val_loss_sum = 0.0
        val_correct = 0.0
        val_total = 0
        iter_val = 0

        pbar_val = tqdm(loader.val_batches(), desc=f"Epoch {epoch+1}/{end_epoch} [val]", leave=False)
        for X_val, Y_val in pbar_val:
            iter_val += 1
            out_val = net.forward(X_val)

            loss_layer = CrossEntropyLoss()
            vloss = loss_layer.forward(out_val, Y_val)
            val_loss_sum += vloss

            for i in range(out_val.shape[0]):
                pred = assign_patches(out_val[i])
                acc = compute_reconstruction_accuracy(
                    X_val[i], pred, Y_val[i], num_patches=num_patches
                )
                val_correct += acc
                val_total += 1

            pbar_val.set_postfix(vloss=f"{vloss:.4f}")

        avg_val_loss = val_loss_sum / max(iter_val, 1)
        val_acc = val_correct / max(val_total, 1)

        # ---- LR decay on plateau ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= lr_patience and net.optimizer.lr > lr_min:
                old_lr = net.optimizer.lr
                net.optimizer.lr = max(net.optimizer.lr * lr_factor, lr_min)
                log.info(f"Reducing LR: {old_lr:.6f} → {net.optimizer.lr:.6f}")
                epochs_no_improve = 0

        # ---- Save checkpoint ----
        ckpt_name = f"epoch_{run_name}_epoch_{epoch}.pkl"
        ckpt_path = os.path.join(run_dir, ckpt_name)
        with open(ckpt_path, "wb") as f:
            pickle.dump(net, f)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_path = os.path.join(run_dir, f"best_{run_name}.pkl")
            with open(best_path, "wb") as f:
                pickle.dump(net, f)

        log.info(
            f"Epoch {epoch+1}/{end_epoch} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
            f"{' ★ best' if is_best else ''} | "
            f"LR: {net.optimizer.lr:.6f} | "
            f"Ckpt: {ckpt_path}"
        )

    log.info(f"Run complete. Best val accuracy: {best_val_acc:.4f}")
    return best_path


if __name__ == "__main__":
    log = setup_logging()

    # # Mount Google Drive (no-op outside Colab)
    # mount_google_drive()

    # # ---- Colab Configuration ----
    # dataset_dir = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32"
    # json_file = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32_cross_val/fold_1.json"

    ## ---- Local Configuration ----
    dataset_dir = "dataset/acv_train_32x32"
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"

    image_size = (32, 32)
    num_patches = 4
    lr_factor = 0.5
    lr_min = 1e-6

    # ======================================================================
    # RUN 1: Adam lr=1e-3, batch=16, 50 epochs, default augmentation
    # ======================================================================
    log.info("=" * 60)
    log.info("RUN 1: Adam lr=1e-3, batch=16, default augmentation")
    log.info("=" * 60)

    run1_name = "run1_adam_b16"
    run1_batch_size = 16
    run1_lr = 1e-3
    run1_epochs = 50
    run1_lr_patience = 10
    run1_resume_ckpt = "checkpoints/run_20260226_052437/best_run2_adam_b32_strong_aug.pkl"
    run1_resume_epoch = 202

    timestamp1 = datetime.now().strftime("%Y%m%d_%H%M%S")
    run1_dir = os.path.join("checkpoints", f"run_{timestamp1}")

    loader1 = PatchShuffleDataLoader(
        json_file, dataset_dir,
        batch_size=run1_batch_size, image_size=image_size,
        num_patches=num_patches, shuffle=True
    )

    if run1_resume_ckpt and os.path.exists(run1_resume_ckpt):
        log.info(f"Resuming from checkpoint: {run1_resume_ckpt}")
        with open(run1_resume_ckpt, "rb") as f:
            net = pickle.load(f)
        net.optimizer = Adam(lr=run1_lr)
        start_epoch = run1_resume_epoch
    else:
        net = build_network(lr=run1_lr)
        start_epoch = 0

    end_epoch = start_epoch + run1_epochs
    log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
    log.info(f"Training epochs {start_epoch + 1} → {end_epoch}")

    best_ckpt_1 = train_run(
        net, loader1, start_epoch, end_epoch, run1_dir, run1_name,
        use_augmentation=True, num_patches=num_patches,
        lr_patience=run1_lr_patience, lr_factor=lr_factor, lr_min=lr_min,
        log=log, aug_params=None
    )

    # ======================================================================
    # RUN 2: Adam lr=5e-4, batch=32, 40 epochs, stronger augmentation
    # ======================================================================
    log.info("=" * 60)
    log.info("RUN 2: Adam lr=5e-4, batch=32, stronger augmentation")
    log.info("=" * 60)

    run2_name = "run2_adam_b32_strong_aug"
    run2_batch_size = 32
    run2_lr = 5e-4
    run2_epochs = 40
    run2_lr_patience = 8
    run2_aug_params = {
        "max_rotation": 15,
        "brightness_range": 0.3,
        "contrast_range": 0.3,
        "blur_prob": 0.5,
    }

    timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    run2_dir = os.path.join("checkpoints", f"run_{timestamp2}")

    loader2 = PatchShuffleDataLoader(
        json_file, dataset_dir,
        batch_size=run2_batch_size, image_size=image_size,
        num_patches=num_patches, shuffle=True
    )

    if best_ckpt_1 and os.path.exists(best_ckpt_1):
        log.info(f"Loading Run 1 best checkpoint: {best_ckpt_1}")
        with open(best_ckpt_1, "rb") as f:
            net2 = pickle.load(f)
        net2.optimizer = Adam(lr=run2_lr)
    else:
        log.info("No Run 1 best checkpoint found, building fresh network")
        net2 = build_network(lr=run2_lr)

    run2_start = end_epoch
    run2_end = run2_start + run2_epochs
    log.info(f"Training epochs {run2_start + 1} → {run2_end}")

    best_ckpt_2 = train_run(
        net2, loader2, run2_start, run2_end, run2_dir, run2_name,
        use_augmentation=True, num_patches=num_patches,
        lr_patience=run2_lr_patience, lr_factor=lr_factor, lr_min=lr_min,
        log=log, aug_params=run2_aug_params
    )

    log.info("=" * 60)
    log.info("All runs complete.")
    log.info(f"Run 1 best checkpoint: {best_ckpt_1}")
    log.info(f"Run 2 best checkpoint: {best_ckpt_2}")
    log.info("=" * 60)
