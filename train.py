"""
Network builder and training loop. Experiment scripts in experiments/ import from here.
Augmentation functions live in utils.py.
"""
import os
import pickle
from tqdm import tqdm
from layers import (
    Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer, CrossEntropyLoss, Adam
)
from utils import assign_patches, compute_reconstruction_accuracy, augmented_train_batches


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
