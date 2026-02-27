"""Base run: SGD lr=1e-3, batch=4, no augmentation.

The very first training configuration. Uses the baseline CNN architecture
with SGD optimizer. Reached ~49% val accuracy before switching to Adam.
"""
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from layers import Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer, SGD
from utils import PatchShuffleDataLoader, compute_total_receptive_field, setup_logging
from train import train_run


def build_network(lr=1e-3):
    """Baseline patch-sorting CNN with SGD optimizer."""
    return Network([
        ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        MaxPoolLayer(size=2, stride=2),
        ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        MaxPoolLayer(size=2, stride=2),
        ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        MaxPoolLayer(size=2, stride=2),
        ConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
        ConvLayer(in_channels=64, out_channels=16, kernel_size=1, stride=1, pad=0),
        SoftmaxLayer(axis=1),
    ], optimizer=SGD(lr=lr))


if __name__ == "__main__":
    log = setup_logging(log_file="logs/train_base.log")

    dataset_dir = "dataset/acv_train_32x32"
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"

    image_size = (32, 32)
    num_patches = 4

    run_name = "base_sgd"
    batch_size = 4
    lr = 1e-3
    epochs = 30
    lr_patience = 10
    lr_factor = 0.5
    lr_min = 1e-6

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"run_base_{timestamp}")

    loader = PatchShuffleDataLoader(
        json_file, dataset_dir,
        batch_size=batch_size, image_size=image_size,
        num_patches=num_patches, shuffle=True
    )

    net = build_network(lr=lr)

    log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
    log.info(f"Architecture: Baseline CNN + SGD")
    log.info(f"Training epochs 1 → {epochs} | batch={batch_size} | lr={lr}")

    best_ckpt = train_run(
        net, loader, start_epoch=0, end_epoch=epochs,
        run_dir=run_dir, run_name=run_name,
        use_augmentation=False, num_patches=num_patches,
        lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min,
        log=log
    )

    log.info(f"Done. Best checkpoint: {best_ckpt}")
