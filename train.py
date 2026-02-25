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
                  brightness_range=0.2, contrast_range=0.2):
    """Apply random augmentations to a grayscale image (H, W, C) uint8."""
    h, w = image.shape[:2]

    if random.random() < p_flip:
        image = np.ascontiguousarray(image[:, ::-1, :])

    angle = random.uniform(-max_rotation, max_rotation)
    if abs(angle) > 0.5:
        M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv.warpAffine(image, M, (w, h), borderMode=cv.BORDER_REFLECT_101)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

    shift = random.uniform(-brightness_range, brightness_range) * 255
    image = np.clip(image.astype(np.float32) + shift, 0, 255).astype(np.uint8)

    factor = random.uniform(1.0 - contrast_range, 1.0 + contrast_range)
    mean = image.mean()
    image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        image = cv.GaussianBlur(image, (ksize, ksize), 0)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

    return image


def augmented_train_batches(loader):
    """Yield augmented training batches. Augmentation runs before patch shuffle."""
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

            image = augment_image(image)

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

    # ---- Run / resume configuration ----
    run_name = "stage2_2x2_10k_aug"
    batch_size = 16
    image_size = (32, 32)
    num_patches = 4
    additional_epochs = 50
    lr = 1e-3
    use_augmentation = True

    # LR decay: halve LR if val loss doesn't improve for `lr_patience` epochs
    lr_patience = 10
    lr_factor = 0.5
    lr_min = 1e-6

    resume_checkpoint = "checkpoints/run_20260225_152253/epoch_stage2_2x2_10k_aug_epoch_110.pkl"
    resume_epoch = 111

    # ---- Run directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # ---- Single loader (used for both train + val) ----
    loader = PatchShuffleDataLoader(
        json_file, dataset_dir,
        batch_size=batch_size, image_size=image_size,
        num_patches=num_patches, shuffle=True
    )

    # ---- Load or build network ----
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        log.info(f"Resuming from checkpoint: {resume_checkpoint}")
        with open(resume_checkpoint, "rb") as f:
            net = pickle.load(f)
        # Switch optimizer to Adam when resuming from SGD-trained checkpoint
        net.optimizer = Adam(lr=lr)
        start_epoch = resume_epoch
        log.info(f"Loaded model — continuing from epoch {start_epoch + 1}")
    else:
        net = build_network(lr=lr)
        start_epoch = 0
        log.info("No checkpoint found — training from scratch")

    end_epoch = start_epoch + additional_epochs
    log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
    log.info(f"Training epochs {start_epoch + 1} → {end_epoch} | "
             f"batch_size={batch_size} | augmentation={use_augmentation}")

    # ---- Training loop ----
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(start_epoch, end_epoch):
        # ---- Train ----
        epoch_loss = 0.0
        iter_train = 0

        if use_augmentation:
            train_iter = augmented_train_batches(loader)
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

    log.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
