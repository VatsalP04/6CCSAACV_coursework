"""Fine-tune global context model with morphological + noise augmentation
for better generalisation on unseen datasets.

Uses the GlobalContextLayer (SE blocks) architecture which performed best,
and adds erosion/dilation + noise augmentation to handle varying stroke
thicknesses and pixel distributions.
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
    GlobalContextLayer, Adam
)
from utils import (
    PatchShuffleDataLoader, compute_total_receptive_field,
    setup_logging
)
from train import train_run


if __name__ == "__main__":
    log = setup_logging(log_file="logs/train_generalise.log")

    dataset_dir = "dataset/acv_train_32x32"
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"

    image_size = (32, 32)
    num_patches = 4

    run_name = "generalise"
    batch_size = 16
    lr = 5e-4
    epochs = 80
    lr_patience = 10
    lr_factor = 0.5
    lr_min = 1e-6

    # Augmentation with morphological transforms + noise for generalisation
    aug_params = {
        "max_rotation": 15,
        "brightness_range": 0.4,
        "contrast_range": 0.4,
        "blur_prob": 0.4,
        "morph_prob": 0.5,
        "noise_prob": 0.3,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"run_gen_{timestamp}")

    loader = PatchShuffleDataLoader(
        json_file, dataset_dir,
        batch_size=batch_size, image_size=image_size,
        num_patches=num_patches, shuffle=True
    )

    # Resume from best global context checkpoint
    resume_ckpt = "checkpoints/run_gc_20260226_185216/best_global_context.pkl"

    if not os.path.exists(resume_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {resume_ckpt}")

    log.info(f"Resuming from global context model: {resume_ckpt}")
    with open(resume_ckpt, "rb") as f:
        net = pickle.load(f)
    net.optimizer = Adam(lr=lr)

    start_epoch = 0
    end_epoch = epochs

    log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
    log.info(f"Architecture: Global Context (SE blocks) + generalisation augmentation")
    log.info(f"Augmentation: morph_prob=0.5, noise_prob=0.3, "
             f"brightness=0.4, contrast=0.4, rotation=15, blur=0.4")
    log.info(f"Training epochs 1 → {epochs} | batch={batch_size} | lr={lr}")

    best_ckpt = train_run(
        net, loader, start_epoch, end_epoch, run_dir, run_name,
        use_augmentation=True, num_patches=num_patches,
        lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min,
        log=log, aug_params=aug_params
    )

    log.info(f"Done. Best checkpoint: {best_ckpt}")
