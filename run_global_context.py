"""Train with GlobalContextLayer added after the bottleneck convolutions."""
import os
import pickle
from datetime import datetime
from layers import (
    Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer,
    GlobalContextLayer, Adam
)
from utils import (
    PatchShuffleDataLoader, compute_total_receptive_field,
    setup_logging
)
from train import train_run


def build_network_global_context(lr=1e-3):
    """CNN with global context (squeeze-and-excite) after each pooling stage."""
    return Network([
        ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),       # 32x32 -> 16x16
        GlobalContextLayer(32, reduction=4),   # squeeze-excite on 32 channels

        ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),       # 16x16 -> 8x8
        GlobalContextLayer(64, reduction=4),   # squeeze-excite on 64 channels

        ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),        # 8x8 -> 4x4
        GlobalContextLayer(128, reduction=4),   # squeeze-excite on 128 channels

        ConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),

        ConvLayer(in_channels=64, out_channels=16, kernel_size=1, stride=1, pad=0),

        SoftmaxLayer(axis=1),
    ], optimizer=Adam(lr=lr))


if __name__ == "__main__":
    log = setup_logging(log_file="train_global_context.log")

    dataset_dir = "dataset/acv_train_32x32"
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"

    image_size = (32, 32)
    num_patches = 4

    run_name = "global_context"
    batch_size = 16
    lr = 1e-3
    epochs = 100
    lr_patience = 10
    lr_factor = 0.5
    lr_min = 1e-6

    aug_params = {
        "max_rotation": 15,
        "brightness_range": 0.3,
        "contrast_range": 0.3,
        "blur_prob": 0.5,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"run_gc_{timestamp}")

    loader = PatchShuffleDataLoader(
        json_file, dataset_dir,
        batch_size=batch_size, image_size=image_size,
        num_patches=num_patches, shuffle=True
    )

    net = build_network_global_context(lr=lr)
    log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
    log.info(f"Architecture: Global Context (SE blocks after each pool)")
    log.info(f"Training epochs 1 → {epochs} | batch={batch_size} | lr={lr}")

    best_ckpt = train_run(
        net, loader, start_epoch=0, end_epoch=epochs,
        run_dir=run_dir, run_name=run_name,
        use_augmentation=True, num_patches=num_patches,
        lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min,
        log=log, aug_params=aug_params
    )

    log.info(f"Done. Best checkpoint: {best_ckpt}")
