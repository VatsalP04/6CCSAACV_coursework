"""
Evaluate a model on unseen data (QMNIST or USPS).

Auto-detects format based on file extension:
  - .gz  → QMNIST IDX3 format (28x28 uint8)
  - .h5  → USPS HDF5 format (16x16 float32 in [0,1])

Usage:
    python eval_unseen.py --model final.pkl --data dataset/unseen_data/qmnist-test-images-idx3-ubyte.gz
    python eval_unseen.py --model final.pkl --data dataset/unseen_data/usps.h5
    python eval_unseen.py --model final.pkl --data dataset/unseen_data/usps.h5 --num-images 2007
"""
import argparse
import gzip
import os
import sys
import struct
import pickle
import numpy as np
import cv2 as cv
from tqdm import tqdm

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from layers import (
    Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer,
    SGD, Adam, GlobalContextLayer, DropoutLayer
)
from utils import assign_patches, compute_reconstruction_accuracy


def load_qmnist_images(path):
    """Load images from IDX3 gzip file. Returns (N, H, W) uint8 array."""
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def load_usps_images(path, split='test'):
    """Load images from USPS HDF5 file. Returns (N, H, W) uint8 array."""
    import h5py
    with h5py.File(path, 'r') as f:
        data = f[split]['data'][:]  # (N, 256) float32 in [0,1]
    # Reshape to 16x16 and convert to uint8
    images = (data.reshape(-1, 16, 16) * 255).astype(np.uint8)
    return images


def load_dir_images(path, max_images=None):
    """Load grayscale PNGs from a directory (recursive). Returns (N, H, W) uint8 array.

    Handles RGBA images where digit data lives in the alpha channel.
    """
    import glob
    files = sorted(glob.glob(os.path.join(path, '**', '*.png'), recursive=True))
    if max_images:
        files = files[:max_images]
    images = []
    for f in files:
        img = cv.imread(f, cv.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            # RGBA: digit data is in the alpha channel
            img = img[:, :, 3]
        elif img.ndim == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        images.append(img)
    return np.stack(images, axis=0)


def load_images(path):
    """Auto-detect format and load images as (N, H, W) uint8."""
    if path.endswith('.gz'):
        return load_qmnist_images(path), 'QMNIST'
    elif path.endswith('.h5'):
        return load_usps_images(path), 'USPS'
    elif os.path.isdir(path):
        return load_dir_images(path), os.path.basename(path.rstrip('/'))
    else:
        raise ValueError(f"Unknown format: {path}. Expected .gz, .h5, or directory")


def shuffle_patches(image, num_patches=4):
    """Shuffle patches of a grayscale image (H, W)."""
    H, W = image.shape
    N = num_patches
    ph, pw = H // N, W // N
    num_total = N * N

    indices = np.random.permutation(num_total)
    patches = image.reshape(N, ph, N, pw).transpose(0, 2, 1, 3).reshape(num_total, ph, pw)
    shuffled_patches = patches[indices]
    shuffled = shuffled_patches.reshape(N, N, ph, pw).transpose(0, 2, 1, 3).reshape(H, W)

    label = np.empty(num_total, dtype=np.int64)
    label[indices] = np.arange(num_total, dtype=np.int64)

    return shuffled, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model pickle file")
    parser.add_argument("--data", default="dataset/unseen_data/qmnist-test-images-idx3-ubyte.gz",
                        help="Path to dataset (.gz for QMNIST, .h5 for USPS)")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-patches", type=int, default=4)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load model
    print(f"Loading model: {args.model}")
    with open(args.model, "rb") as f:
        net = pickle.load(f)
    if hasattr(net, 'eval_mode'):
        net.eval_mode()

    # Load images (auto-detect format)
    print(f"Loading images: {args.data}")
    all_images, dataset_name = load_images(args.data)
    print(f"Dataset: {dataset_name} | Total images: {len(all_images)} | "
          f"Image size: {all_images.shape[1]}x{all_images.shape[2]}")

    # Select subset
    num_eval = min(args.num_images, len(all_images))
    indices = np.random.choice(len(all_images), num_eval, replace=False)
    images = all_images[indices]
    print(f"Evaluating on {num_eval} images\n")

    N = args.num_patches
    total_acc = 0.0
    num_evaluated = 0

    pbar = tqdm(range(0, num_eval, args.batch_size), desc="Evaluating")
    for start in pbar:
        end = min(start + args.batch_size, num_eval)
        batch_X = []
        batch_Y = []

        for i in range(start, end):
            img = images[i]
            # Resize to 32x32
            img_resized = cv.resize(img, (32, 32))

            shuffled, label = shuffle_patches(img_resized, num_patches=N)

            img_float = shuffled.astype(np.float32) / 255.0
            img_chw = img_float[np.newaxis, :, :]  # (1, 32, 32)

            batch_X.append(img_chw)
            batch_Y.append(label)

        X = np.stack(batch_X, axis=0)
        Y = np.stack(batch_Y, axis=0)

        out = net.forward(X)

        for i in range(out.shape[0]):
            pred = assign_patches(out[i])
            acc = compute_reconstruction_accuracy(
                X[i], pred, Y[i], num_patches=N
            )
            total_acc += acc
            num_evaluated += 1

        pbar.set_postfix(acc=f"{total_acc / num_evaluated:.4f}")

    avg_acc = total_acc / max(num_evaluated, 1)

    print(f"\n{'=' * 50}")
    print(f"UNSEEN DATA EVALUATION — {dataset_name}")
    print(f"{'=' * 50}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Images evaluated: {num_evaluated}")
    print(f"Average accuracy: {avg_acc:.4f} ({avg_acc * 100:.2f}%)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
