"""
Visualise predictions vs ground truth from mock test.

Usage:
    python evaluation/visualise_preds.py --predictions evaluation/predictions.csv --ground-truth evaluation/mock_test_large_gt.csv --test-dir evaluation/mock_test_large --originals dataset/acv_train_32x32 --num-images 8 --output images/mock_test_examples.png
"""
import argparse
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def reconstruct(shuffled, assignment, num_patches=4):
    """Reconstruct image from shuffled using predicted assignment."""
    H, W = shuffled.shape
    N = num_patches
    ph, pw = H // N, W // N

    patches = shuffled.reshape(N, ph, N, pw).transpose(0, 2, 1, 3).reshape(N*N, ph, pw)
    inv = np.argsort(assignment)

    recon = np.zeros_like(shuffled)
    for new_pos in range(N * N):
        orig_id = inv[new_pos]
        oi, oj = orig_id // N, orig_id % N
        recon[oi*ph:(oi+1)*ph, oj*pw:(oj+1)*pw] = patches[new_pos]
    return recon


def compute_accuracy(pred, gt, shuffled_float, num_patches=4, eps=0.05):
    H, W = shuffled_float.shape
    N = num_patches
    ph, pw = H // N, W // N
    correct = 0
    total = 0
    idx = -1
    for y in range(0, H, ph):
        for x in range(0, W, pw):
            idx += 1
            if np.all(shuffled_float[y:y+ph, x:x+pw] < eps):
                continue
            total += 1
            if pred[idx] == gt[idx]:
                correct += 1
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--test-dir", required=True, help="Shuffled images folder")
    parser.add_argument("--originals", default="dataset/acv_train_32x32", help="Original unshuffled images")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--output", default="images/mock_test_examples.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Parse files
    gt = {}
    with open(args.ground_truth) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            gt[parts[0]] = np.array([int(x) for x in parts[1:]])

    preds = {}
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            preds[parts[0]] = np.array([int(x) for x in parts[1:]])

    # Score all and sort
    results = []
    for fname in gt:
        if fname not in preds:
            continue
        img = cv.imread(os.path.join(args.test_dir, fname), cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_float = img.astype(np.float32) / 255.0
        acc = compute_accuracy(preds[fname], gt[fname], img_float)
        results.append((fname, acc))

    results.sort(key=lambda x: x[1], reverse=True)

    # Pick best half + worst half
    n = min(args.num_images, len(results))
    n_good = n // 2
    n_bad = n - n_good
    selected = results[:n_good] + results[-n_bad:]

    # Plot: shuffled | reconstructed (pred) | original
    fig, axes = plt.subplots(len(selected), 3, figsize=(7, 2.2 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Shuffled", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Reconstructed", fontsize=11, fontweight="bold")
    axes[0, 2].set_title("Original", fontsize=11, fontweight="bold")

    for i, (fname, acc) in enumerate(selected):
        shuffled = cv.imread(os.path.join(args.test_dir, fname), cv.IMREAD_GRAYSCALE)
        recon = reconstruct(shuffled, preds[fname])

        # Try to load original
        orig_path = os.path.join(args.originals, fname)
        if os.path.exists(orig_path):
            original = cv.imread(orig_path, cv.IMREAD_GRAYSCALE)
            original = cv.resize(original, (32, 32))
        else:
            # Reconstruct with ground truth as fallback
            original = reconstruct(shuffled, gt[fname])

        axes[i, 0].imshow(shuffled, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon, cmap="gray", vmin=0, vmax=255)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(original, cmap="gray", vmin=0, vmax=255)
        axes[i, 2].axis("off")

        label = f"{acc*100:.0f}%"
        if i < n_good:
            label += " (best)"
        else:
            label += " (worst)"
        axes[i, 2].text(1.05, 0.5, label, transform=axes[i, 2].transAxes,
                        fontsize=9, va="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output} with {len(selected)} examples ({n_good} best, {n_bad} worst)")
    plt.show()


if __name__ == "__main__":
    main()
