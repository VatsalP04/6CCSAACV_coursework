"""Run 1: Adam lr=1e-3, batch=16, 50 epochs, default augmentation.

Resumes from a previous checkpoint if available, otherwise trains from scratch.
"""
import os
import sys
import pickle
from datetime import datetime

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from layers import Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer, Adam
from utils import PatchShuffleDataLoader, compute_total_receptive_field, setup_logging
from train import train_run


def build_network(lr=1e-3):
    """Baseline patch-sorting CNN (Week 5 architecture)."""
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
    ], optimizer=Adam(lr=lr))


if __name__ == "__main__":
    log = setup_logging(log_file="logs/train_run1.log")

    dataset_dir = "dataset/acv_train_32x32"
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"

    image_size = (32, 32)
    num_patches = 4
    lr_factor = 0.5
    lr_min = 1e-6

    run_name = "run1_adam_b16"
    batch_size = 16
    lr = 1e-3
    epochs = 50
    lr_patience = 10
    resume_ckpt = "checkpoints/run_20260226_052437/best_run2_adam_b32_strong_aug.pkl"
    resume_epoch = 202

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"run_{timestamp}")

    loader = PatchShuffleDataLoader(
        json_file, dataset_dir,
        batch_size=batch_size, image_size=image_size,
        num_patches=num_patches, shuffle=True
    )

    if resume_ckpt and os.path.exists(resume_ckpt):
        log.info(f"Resuming from checkpoint: {resume_ckpt}")
        with open(resume_ckpt, "rb") as f:
            net = pickle.load(f)
        net.optimizer = Adam(lr=lr)
        start_epoch = resume_epoch
    else:
        net = build_network(lr=lr)
        start_epoch = 0

    end_epoch = start_epoch + epochs
    log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
    log.info(f"Training epochs {start_epoch + 1} → {end_epoch}")

    best_ckpt = train_run(
        net, loader, start_epoch, end_epoch, run_dir, run_name,
        use_augmentation=True, num_patches=num_patches,
        lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min,
        log=log, aug_params=None
    )

    log.info(f"Done. Best checkpoint: {best_ckpt}")
