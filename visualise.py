"""
Generate visual examples of shuffled → reconstructed → original images.
Saves a grid figure for the report showing good and bad reconstructions.

Usage:
    python visualise.py --model final.pkl --num-images 8
"""
import argparse
import json
import os
import pickle
import random

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.optimize

import gzip
import struct

from layers import Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer, SGD, Adam, GlobalContextLayer, DropoutLayer


def assign_patches(probs):
    C, H, W = probs.shape
    cost = -probs.transpose(1, 2, 0).reshape(-1, C)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    assignment = np.empty(C, dtype=np.int64)
    assignment[row_ind] = col_ind
    return assignment


def shuffle_patches(image, num_patches=4):
    """Shuffle patches of a grayscale (H,W) image. Returns shuffled, gt_label."""
    H, W = image.shape
    N = num_patches
    ph, pw = H // N, W // N
    num_total = N * N

    patches = image.reshape(N, ph, N, pw).transpose(0, 2, 1, 3).reshape(num_total, ph, pw)
    indices = np.random.permutation(num_total)
    shuffled_patches = patches[indices]
    shuffled = shuffled_patches.reshape(N, N, ph, pw).transpose(0, 2, 1, 3).reshape(H, W)

    label = np.empty(num_total, dtype=np.int64)
    label[indices] = np.arange(num_total, dtype=np.int64)

    return shuffled, label, indices


def reconstruct_image(shuffled, pred_assignment, num_patches=4):
    """Rearrange patches of a shuffled image using predicted assignment.

    pred_assignment follows the label convention: pred[original_id] = new_pos.
    To reconstruct, we invert: inv[new_pos] = original_id, then place each
    shuffled patch (at new_pos) back into its original_id location.
    """
    H, W = shuffled.shape
    N = num_patches
    ph, pw = H // N, W // N
    num_total = N * N

    patches = shuffled.reshape(N, ph, N, pw).transpose(0, 2, 1, 3).reshape(num_total, ph, pw)

    # Invert: inv[new_pos] = original_id
    inv = np.argsort(pred_assignment)

    recon = np.zeros_like(shuffled)
    for new_pos in range(num_total):
        orig_id = inv[new_pos]
        oi, oj = orig_id // N, orig_id % N
        recon[oi*ph:(oi+1)*ph, oj*pw:(oj+1)*pw] = patches[new_pos]

    return recon


def compute_accuracy(pred, gt, shuffled_float, num_patches=4, eps=0.05):
    """Accuracy excluding black patches."""
    H, W = shuffled_float.shape
    N = num_patches
    ph, pw = H // N, W // N
    num_correct = 0
    num_total = 0
    idx = -1
    for y in range(0, H, ph):
        for x in range(0, W, pw):
            idx += 1
            if np.all(shuffled_float[y:y+ph, x:x+pw] < eps):
                continue
            num_total += 1
            if pred[idx] == gt[idx]:
                num_correct += 1
    return num_correct / max(num_total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", default="dataset/acv_train_32x32")
    parser.add_argument("--json-file", default="dataset/acv_train_32x32_cross_val/fold_1.json")
    parser.add_argument("--unseen", default=None,
                        help="Path to unseen data file (.gz for QMNIST, .h5 for USPS)")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="images/report_examples.png")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    with open(args.model, "rb") as f:
        net = pickle.load(f)
    if hasattr(net, 'eval_mode'):
        net.eval_mode()

    # Load images from unseen data or val set
    if args.unseen:
        if args.unseen.endswith('.gz'):
            with gzip.open(args.unseen, 'rb') as f:
                magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
                all_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
            tag = 'qmnist'
        elif args.unseen.endswith('.h5'):
            import h5py
            with h5py.File(args.unseen, 'r') as f:
                data = f['test']['data'][:]
            all_imgs = (data.reshape(-1, 16, 16) * 255).astype(np.uint8)
            tag = 'usps'
        elif os.path.isdir(args.unseen):
            import glob as globmod
            files = sorted(globmod.glob(os.path.join(args.unseen, '**', '*.png'), recursive=True))
            all_imgs = []
            for fp in files:
                im = cv.imread(fp, cv.IMREAD_UNCHANGED)
                if im is None:
                    continue
                if im.ndim == 3 and im.shape[2] == 4:
                    im = im[:, :, 3]  # RGBA: digit in alpha channel
                elif im.ndim == 3:
                    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                all_imgs.append(im)
            all_imgs = np.stack(all_imgs, axis=0)
            tag = os.path.basename(args.unseen.rstrip('/'))
        else:
            raise ValueError(f"Unknown format: {args.unseen}")
        idxs = np.random.choice(len(all_imgs), min(args.num_images * 3, len(all_imgs)), replace=False)
        image_source = [(f"{tag}_{i}", cv.resize(all_imgs[i], (32, 32))) for i in idxs]
    else:
        with open(args.json_file, "r") as f:
            val_files = json.load(f)["val"]
        random.shuffle(val_files)
        image_source = []
        for fname in val_files:
            if len(image_source) >= args.num_images * 3:
                break
            img = cv.imread(os.path.join(args.dataset_dir, fname))
            if img is None:
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (32, 32))
            image_source.append((fname, img))

    # Process images and collect results
    results = []
    for fname, img in image_source:
        if len(results) >= args.num_images * 3:
            break

        original = img.copy()
        shuffled, gt_label, indices = shuffle_patches(img, num_patches=4)

        # Forward pass
        inp = shuffled.astype(np.float32) / 255.0
        inp_batch = inp[np.newaxis, np.newaxis, :, :]  # (1, 1, 32, 32)
        probs = net.forward(inp_batch)[0]  # (16, 4, 4)
        pred = assign_patches(probs)

        acc = compute_accuracy(pred, gt_label, inp, num_patches=4)
        recon = reconstruct_image(shuffled, pred, num_patches=4)

        results.append({
            "original": original,
            "shuffled": shuffled,
            "reconstructed": recon,
            "accuracy": acc,
            "filename": fname,
        })

    # Sort by accuracy: best first, worst last
    results.sort(key=lambda r: r["accuracy"], reverse=True)

    # Pick top half as good, bottom half as bad
    n = min(args.num_images, len(results))
    n_good = n // 2
    n_bad = n - n_good
    selected = results[:n_good] + results[-n_bad:]

    # Plot grid: rows = images, cols = shuffled | reconstructed | original
    fig, axes = plt.subplots(len(selected), 3, figsize=(6, 2 * len(selected)))

    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Shuffled", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("Reconstructed", fontsize=10, fontweight="bold")
    axes[0, 2].set_title("Original", fontsize=10, fontweight="bold")

    for i, r in enumerate(selected):
        axes[i, 0].imshow(r["shuffled"], cmap="gray", vmin=0, vmax=255)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(r["reconstructed"], cmap="gray", vmin=0, vmax=255)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(r["original"], cmap="gray", vmin=0, vmax=255)
        axes[i, 2].axis("off")

        # Accuracy label on the right
        acc_pct = r["accuracy"] * 100
        label = f"{acc_pct:.0f}%"
        if i < n_good:
            label += " (good)"
        else:
            label += " (poor)"
        axes[i, 2].text(1.05, 0.5, label, transform=axes[i, 2].transAxes,
                        fontsize=8, va="center")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output} with {len(selected)} examples")
    plt.show()


if __name__ == "__main__":
    main()
