"""Global Context + Dropout architecture for better generalisation.

Adds dropout after each ReLU block to prevent overfitting to training
distribution. Combined with morphological augmentation to handle
varying stroke thicknesses in unseen datasets.
"""
import os
import sys
import pickle
from datetime import datetime

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from layers import (
    Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer,
    GlobalContextLayer, DropoutLayer, Adam
)
from utils import (
    PatchShuffleDataLoader, compute_total_receptive_field,
    setup_logging
)
from train import train_run


def transfer_conv_weights(src_net, dst_net):
    """Copy ConvLayer weights from src to dst (matching by order)."""
    src_convs = [l for l in src_net.layers if isinstance(l, ConvLayer)]
    dst_convs = [l for l in dst_net.layers if isinstance(l, ConvLayer)]
    copied = 0
    for s, d in zip(src_convs, dst_convs):
        if s.W.shape == d.W.shape:
            d.W = s.W.copy()
            d.b = s.b.copy()
            copied += 1
    return copied


def transfer_gc_weights(src_net, dst_net):
    """Copy GlobalContextLayer weights from src to dst (matching by order)."""
    src_gc = [l for l in src_net.layers if isinstance(l, GlobalContextLayer)]
    dst_gc = [l for l in dst_net.layers if isinstance(l, GlobalContextLayer)]
    copied = 0
    for s, d in zip(src_gc, dst_gc):
        if s.W1.shape == d.W1.shape:
            d.W1 = s.W1.copy()
            d.b1 = s.b1.copy()
            d.W2 = s.W2.copy()
            d.b2 = s.b2.copy()
            copied += 1
    return copied


def build_network_gc_dropout(lr=1e-3, drop_p=0.15):
    """Global Context CNN with dropout after conv blocks for generalisation."""
    return Network([
        ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        DropoutLayer(p=drop_p),

        ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),        # 32x32 -> 16x16
        GlobalContextLayer(32, reduction=4),

        ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        DropoutLayer(p=drop_p),

        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),        # 16x16 -> 8x8
        GlobalContextLayer(64, reduction=4),

        ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        DropoutLayer(p=drop_p),

        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),         # 8x8 -> 4x4
        GlobalContextLayer(128, reduction=4),

        ConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        DropoutLayer(p=drop_p),

        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),

        ConvLayer(in_channels=64, out_channels=16, kernel_size=1, stride=1, pad=0),

        SoftmaxLayer(axis=1),
    ], optimizer=Adam(lr=lr))


if __name__ == "__main__":
    log = setup_logging(log_file="logs/train_dropout.log")

    dataset_dir = "dataset/acv_train_32x32"
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"

    image_size = (32, 32)
    num_patches = 4

    run_name = "gc_dropout"
    batch_size = 16
    lr = 5e-4
    epochs = 80
    lr_patience = 10
    lr_factor = 0.5
    lr_min = 1e-6
    drop_p = 0.15  # light dropout — enough to regularise, not too much to slow learning

    # Same generalisation augmentation
    aug_params = {
        "max_rotation": 15,
        "brightness_range": 0.4,
        "contrast_range": 0.4,
        "blur_prob": 0.4,
        "morph_prob": 0.5,
        "noise_prob": 0.3,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"run_drop_{timestamp}")

    loader = PatchShuffleDataLoader(
        json_file, dataset_dir,
        batch_size=batch_size, image_size=image_size,
        num_patches=num_patches, shuffle=True
    )

    # Transfer weights from best global context checkpoint
    pretrained_ckpt = "checkpoints/run_gc_20260226_185216/best_global_context.pkl"

    net = build_network_gc_dropout(lr=lr, drop_p=drop_p)

    if os.path.exists(pretrained_ckpt):
        with open(pretrained_ckpt, "rb") as f:
            old_net = pickle.load(f)
        n_conv = transfer_conv_weights(old_net, net)
        n_gc = transfer_gc_weights(old_net, net)
        log.info(f"Transferred {n_conv} conv + {n_gc} GC layers from {pretrained_ckpt}")
        del old_net
    else:
        log.info("No pretrained checkpoint found — training from scratch")

    log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
    log.info(f"Architecture: Global Context + Dropout(p={drop_p})")
    log.info(f"Augmentation: morph=0.5, noise=0.3, brightness=0.4, contrast=0.4")
    log.info(f"Training epochs 1 → {epochs} | batch={batch_size} | lr={lr}")

    best_ckpt = train_run(
        net, loader, start_epoch=0, end_epoch=epochs,
        run_dir=run_dir, run_name=run_name,
        use_augmentation=True, num_patches=num_patches,
        lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min,
        log=log, aug_params=aug_params
    )

    log.info(f"Done. Best checkpoint: {best_ckpt}")
