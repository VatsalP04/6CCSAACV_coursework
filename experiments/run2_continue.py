"""Continue Run 2 from its best checkpoint for another 40 epochs."""
import os
import sys
import pickle
from datetime import datetime

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from layers import Adam
from utils import (
    PatchShuffleDataLoader, compute_total_receptive_field,
    setup_logging
)
from train import train_run

if __name__ == "__main__":
    log = setup_logging()

    dataset_dir = "dataset/acv_train_32x32"
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"

    image_size = (32, 32)
    num_patches = 4

    # ---- Config ----
    run_name = "run2_continued"
    batch_size = 32
    lr = 2.5e-4
    epochs = 100
    lr_patience = 8
    lr_factor = 0.5
    lr_min = 1e-6
    resume_ckpt = "checkpoints/run_20260226_052437/best_run2_adam_b32_strong_aug.pkl"
    resume_epoch = 197

    aug_params = {
        "max_rotation": 15,
        "brightness_range": 0.3,
        "contrast_range": 0.3,
        "blur_prob": 0.5,
    }

    # ---- Setup ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"run_{timestamp}")

    loader = PatchShuffleDataLoader(
        json_file, dataset_dir,
        batch_size=batch_size, image_size=image_size,
        num_patches=num_patches, shuffle=True
    )

    if not os.path.exists(resume_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {resume_ckpt}")
    log.info(f"Resuming from: {resume_ckpt}")
    with open(resume_ckpt, "rb") as f:
        net = pickle.load(f)
    net.optimizer = Adam(lr=lr)

    start_epoch = resume_epoch
    end_epoch = start_epoch + epochs
    log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
    log.info(f"Training epochs {start_epoch + 1} → {end_epoch} | "
             f"batch={batch_size} | lr={lr} | stronger aug")

    best_ckpt = train_run(
        net, loader, start_epoch, end_epoch, run_dir, run_name,
        use_augmentation=True, num_patches=num_patches,
        lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min,
        log=log, aug_params=aug_params
    )

    log.info(f"Done. Best checkpoint: {best_ckpt}")
