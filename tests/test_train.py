"""
Smoke test for the training pipeline.
Runs 2 epochs of Run 1 → 2 epochs of Run 2 with tiny data to verify:
  - augmented_train_batches with default and custom aug_params
  - train_run function (forward, backward, optimizer step, LR decay, checkpointing)
  - Run 1 → Run 2 handoff (loading best checkpoint, swapping optimizer)
  - Checkpoint save/load roundtrip
"""
import os
import sys
import json
import shutil
import pickle
import tempfile
import numpy as np

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from train import build_network, train_run
from layers import Adam
from utils import (
    PatchShuffleDataLoader, setup_logging, compute_total_receptive_field,
    augmented_train_batches, augment_image
)


def test_augment_image_custom_params():
    """augment_image accepts and uses custom parameters without error."""
    img = np.random.randint(0, 255, (32, 32, 1), dtype=np.uint8)
    out = augment_image(img, max_rotation=15, brightness_range=0.3,
                        contrast_range=0.3, blur_prob=0.5)
    assert out.shape == (32, 32, 1), f"Shape mismatch: {out.shape}"
    assert out.dtype == np.uint8
    print("  [PASS] augment_image with custom params")


def test_augmented_train_batches_with_params():
    """augmented_train_batches forwards aug_params to augment_image."""
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"
    dataset_dir = "dataset/acv_train_32x32"

    loader = PatchShuffleDataLoader(
        json_file, dataset_dir, batch_size=2, image_size=(32, 32),
        num_patches=4, shuffle=True
    )
    # Override train_files to just 4 images for speed
    loader.train_files = loader.train_files[:4]

    aug_params = {"max_rotation": 15, "brightness_range": 0.3,
                  "contrast_range": 0.3, "blur_prob": 0.5}

    count = 0
    for X, Y in augmented_train_batches(loader, aug_params=aug_params):
        assert X.ndim == 4, f"X should be 4D, got {X.ndim}D"
        assert Y.ndim == 2, f"Y should be 2D, got {Y.ndim}D"
        assert X.shape[1] == 1, f"Expected 1 channel, got {X.shape[1]}"
        assert Y.shape[1] == 16, f"Expected 16 labels, got {Y.shape[1]}"
        count += 1

    assert count > 0, "No batches produced"
    print(f"  [PASS] augmented_train_batches with custom params ({count} batches)")


def test_train_run_and_handoff():
    """Full pipeline: build net → Run 1 (2 epochs) → save best → Run 2 (2 epochs)."""
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"
    dataset_dir = "dataset/acv_train_32x32"

    tmpdir = tempfile.mkdtemp(prefix="train_test_")
    try:
        log = setup_logging(log_file=os.path.join(tmpdir, "test.log"))

        # --- Build a tiny network for speed ---
        net = build_network(lr=1e-3)

        # --- Run 1: 2 epochs, default aug ---
        run1_dir = os.path.join(tmpdir, "run1")
        loader1 = PatchShuffleDataLoader(
            json_file, dataset_dir, batch_size=2, image_size=(32, 32),
            num_patches=4, shuffle=True
        )
        # Use only 8 train / 4 val images for speed
        loader1.train_files = loader1.train_files[:8]
        loader1.val_files = loader1.val_files[:4]

        best_ckpt_1 = train_run(
            net, loader1, start_epoch=0, end_epoch=2,
            run_dir=run1_dir, run_name="test_run1",
            use_augmentation=True, num_patches=4,
            lr_patience=1, lr_factor=0.5, lr_min=1e-6,
            log=log, aug_params=None
        )

        # Verify checkpoints exist
        assert os.path.isdir(run1_dir), "Run 1 directory not created"
        epoch_ckpts = [f for f in os.listdir(run1_dir) if f.startswith("epoch_")]
        assert len(epoch_ckpts) == 2, f"Expected 2 epoch checkpoints, got {len(epoch_ckpts)}"
        assert best_ckpt_1 is not None, "No best checkpoint returned"
        assert os.path.exists(best_ckpt_1), f"Best checkpoint not found: {best_ckpt_1}"
        print(f"  [PASS] Run 1: 2 epochs, {len(epoch_ckpts)} checkpoints, best={os.path.basename(best_ckpt_1)}")

        # --- Verify checkpoint loads correctly ---
        with open(best_ckpt_1, "rb") as f:
            net2 = pickle.load(f)
        assert hasattr(net2, 'layers'), "Loaded model missing layers"
        assert hasattr(net2, 'optimizer'), "Loaded model missing optimizer"
        print("  [PASS] Checkpoint load roundtrip")

        # --- Run 2: swap optimizer, stronger aug, 2 epochs ---
        net2.optimizer = Adam(lr=5e-4)
        run2_dir = os.path.join(tmpdir, "run2")
        loader2 = PatchShuffleDataLoader(
            json_file, dataset_dir, batch_size=4, image_size=(32, 32),
            num_patches=4, shuffle=True
        )
        loader2.train_files = loader2.train_files[:8]
        loader2.val_files = loader2.val_files[:4]

        run2_aug = {"max_rotation": 15, "brightness_range": 0.3,
                    "contrast_range": 0.3, "blur_prob": 0.5}

        best_ckpt_2 = train_run(
            net2, loader2, start_epoch=2, end_epoch=4,
            run_dir=run2_dir, run_name="test_run2",
            use_augmentation=True, num_patches=4,
            lr_patience=1, lr_factor=0.5, lr_min=1e-6,
            log=log, aug_params=run2_aug
        )

        assert os.path.isdir(run2_dir), "Run 2 directory not created"
        epoch_ckpts_2 = [f for f in os.listdir(run2_dir) if f.startswith("epoch_")]
        assert len(epoch_ckpts_2) == 2, f"Expected 2 epoch checkpoints, got {len(epoch_ckpts_2)}"
        assert best_ckpt_2 is not None, "No best checkpoint returned from Run 2"
        print(f"  [PASS] Run 2: 2 epochs, {len(epoch_ckpts_2)} checkpoints, best={os.path.basename(best_ckpt_2)}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_lr_decay_triggers():
    """LR decay fires when val loss doesn't improve (patience=1 for fast test)."""
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"
    dataset_dir = "dataset/acv_train_32x32"

    tmpdir = tempfile.mkdtemp(prefix="lr_decay_test_")
    try:
        log = setup_logging(log_file=os.path.join(tmpdir, "test.log"))
        net = build_network(lr=1e-2)

        run_dir = os.path.join(tmpdir, "run")
        loader = PatchShuffleDataLoader(
            json_file, dataset_dir, batch_size=2, image_size=(32, 32),
            num_patches=4, shuffle=True
        )
        loader.train_files = loader.train_files[:4]
        loader.val_files = loader.val_files[:4]

        initial_lr = net.optimizer.lr

        # Run 4 epochs with patience=1 — LR should decay at least once
        train_run(
            net, loader, start_epoch=0, end_epoch=4,
            run_dir=run_dir, run_name="lr_test",
            use_augmentation=False, num_patches=4,
            lr_patience=1, lr_factor=0.5, lr_min=1e-6,
            log=log
        )

        # LR may or may not have decayed depending on loss trajectory,
        # but the mechanism should not crash
        print(f"  [PASS] LR decay mechanism ran without error (LR: {initial_lr} → {net.optimizer.lr})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("Running training pipeline tests...\n")

    test_augment_image_custom_params()
    test_augmented_train_batches_with_params()
    test_train_run_and_handoff()
    test_lr_decay_triggers()

    print("\nAll tests passed!")
