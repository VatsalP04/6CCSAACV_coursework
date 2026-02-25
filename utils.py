import os
import json
import random
import logging
import pickle
from typing import Tuple, List

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.optimize

# Google Colab guard — allows the file to be imported outside Colab
try:
    from google.colab import drive as _colab_drive
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False


def mount_google_drive(drive_path="/content/drive"):
    """Check if Google Drive is mounted in Colab. Mount it if not."""
    if not _IN_COLAB:
        print("Not running in Colab, skipping drive mount.")
        return
    if os.path.exists(drive_path):
        print("Google Drive is already mounted.")
    else:
        print("Mounting Google Drive...")
        _colab_drive.mount(drive_path)
        if os.path.exists(drive_path):
            print("Google Drive mounted successfully.")
        else:
            print("Failed to mount Google Drive. Please try again.")

def setup_logging(log_file="train.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(net, epoch, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pkl")
    with open(path, "wb") as f:
        pickle.dump(net, f)
    return path

def list_all_images(file_path):
    """List all image filenames in a directory."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Directory not found: {file_path}")

    images = []
    for filename in os.listdir(file_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(filename)

    if len(images) == 0:
        print("No images found in the directory.")

    return images

def cross_val_imgs(SEED, K, file_path, output_jsons):
    os.makedirs(output_jsons, exist_ok=True)

    all_images = list_all_images(file_path)
    if len(all_images) < K:
        raise ValueError(f"Need at least {K} images, found {len(all_images)}")

    random.seed(SEED)
    random.shuffle(all_images)

    n = len(all_images)
    fold_size = n // K

    for i in range(K):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < K - 1 else n

        val_files = all_images[val_start:val_end]
        train_files = all_images[:val_start] + all_images[val_end:]

        fold_data = {"train": train_files, "val": val_files}

        fold_path = os.path.join(output_jsons, f"fold_{i+1}.json")
        with open(fold_path, "w") as f:
            json.dump(fold_data, f, indent=4)

        print("Saved:", fold_path)

# cross_val_imgs(SEED = 2025, K = 5, file_path = "dataset/acv_train_32x32", output_jsons= "dataset/acv_train_32x32_cross_val")
# print(os.listdir("dataset/acv_train_32x32_cross_val"))

class PatchShuffleDataLoader:
    def __init__(self, json_file: str, dataset_dir: str, batch_size: int = 8,
                 image_size: Tuple[int, int] = (32, 32), num_patches: int = 4,
                 shuffle: bool = True):
        """
        DataLoader with patch shuffling.
        Labels are permutation vectors of length num_patches^2.

        Args:
            json_file (str): Path to JSON file (contains "train" and "val" lists).
            dataset_dir (str): Root dataset directory containing the images.
            batch_size (int): Number of samples per batch (default=8).
            image_size (Tuple[int,int]): Resize images to this size (W, H). Default (32, 32).
            num_patches (int): Number of patches along one dimension (default=4).
            shuffle (bool): Shuffle dataset each epoch if True (default=True).
        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_patches = num_patches
        self.shuffle = shuffle

        if not os.path.exists(json_file):
            output_jsons = os.path.dirname(json_file)
            print(f"Fold file not found, generating cross-validation splits in {output_jsons} ...")
            cross_val_imgs(SEED=2025, K=5, file_path=dataset_dir, output_jsons=output_jsons)

        with open(json_file, 'r') as f:
            data = json.load(f)
        self.train_files = data["train"]
        self.val_files = data["val"]

        self.patch_size = None  # set on first use

    def _shuffle_patches(self, image: np.ndarray, indices: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised patch shuffle using reshape/transpose instead of loops.

        Args:
            image: (H, W, C)
            indices: optional predefined permutation

        Returns:
            shuffled_image: (H, W, C)
            label: (N*N,) where label[original_patch_id] = new_position
        """
        H, W, C = image.shape
        N = self.num_patches

        if H % N != 0 or W % N != 0:
            raise ValueError(
                f"Image size ({H}, {W}) not divisible by num_patches={N}. "
                "Choose a different num_patches or resize/crop the image."
            )

        ph, pw = H // N, W // N
        self.patch_size = (ph, pw)

        num_total = N * N
        if indices is None:
            indices = np.random.permutation(num_total)
        else:
            indices = np.asarray(indices)
            if indices.shape != (num_total,):
                raise ValueError(f"indices must have shape ({num_total},), got {indices.shape}")
            if set(indices.tolist()) != set(range(num_total)):
                raise ValueError("indices must be a permutation of 0..N^2-1 (no repeats, none missing)")

        # Split into patches: (H,W,C) -> (N,ph,N,pw,C) -> (N*N,ph,pw,C)
        patches = image.reshape(N, ph, N, pw, C).transpose(0, 2, 1, 3, 4).reshape(num_total, ph, pw, C)

        # Reorder patches and stitch back: (N*N,ph,pw,C) -> (H,W,C)
        shuffled_patches = patches[indices]
        shuffled_image = shuffled_patches.reshape(N, N, ph, pw, C).transpose(0, 2, 1, 3, 4).reshape(H, W, C)

        # label[original_id] = new_pos
        label = np.empty(num_total, dtype=np.int64)
        label[indices] = np.arange(num_total, dtype=np.int64)

        return shuffled_image, label

    def _load_image(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess a single image with patch shuffling (CHW + permutation labels).
        """
        full_path = os.path.join(self.dataset_dir, filepath)
        image = cv.imread(full_path)

        if image is None:
            print(f"[Warning] Failed to load image: {full_path}. Skipping...")
            return None, None

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, self.image_size)

        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        image, labels = self._shuffle_patches(image)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        return image, labels

    def _get_batches(self, file_list: List[str]):
        """
        Yield mini-batches of data and patch permutation labels.

        Args:
            file_list (List[str]): List of image filenames to process.

        Yields:
            Tuple[np.ndarray, np.ndarray]:
                - Batch of images: shape (B, C, H, W)
                - Batch of labels: shape (B, N^2) where N is num_patches
        """
        files = list(file_list)
        if self.shuffle:
            random.shuffle(files)

        for start in range(0, len(files), self.batch_size):
            batch_files = files[start:start + self.batch_size]

            X, Y = [], []
            for fname in batch_files:
                img, label = self._load_image(fname)
                if img is None:
                    continue
                X.append(img)
                Y.append(label)

            if X:
                yield np.stack(X, axis=0), np.stack(Y, axis=0)

    def train_batches(self):
        return self._get_batches(self.train_files)

    def val_batches(self):
        return self._get_batches(self.val_files)

    def display_training_image(self, image: np.ndarray):
        new_im = np.transpose(image, (1, 2, 0))
        new_im = np.clip(new_im * 255.0, 0, 255).astype(np.uint8)
        plt.imshow(new_im.squeeze(), cmap="gray")
        plt.axis("off")
        plt.show()

    def display_reconstructed_training_image(self, image: np.ndarray, label: np.ndarray):
        """
        label is original_id -> new_pos.
        To reconstruct, we need the inverse: new_pos -> original_id = argsort(label).
        """
        new_im = np.transpose(image, (1, 2, 0))
        new_im = np.clip(new_im * 255.0, 0, 255).astype(np.uint8)
        inv = np.argsort(label)
        recon, _ = self._shuffle_patches(new_im, inv)
        plt.imshow(recon.squeeze(), cmap="gray")
        plt.axis("off")
        plt.show()


def assign_patches(probs: np.ndarray) -> np.ndarray:
    """
    Hungarian assignment on predicted probabilities.

    Args:
        probs: (C, H, W) with C = H*W

    Returns:
        (C,) assignment where assignment[row] = col
    """
    C, H, W = probs.shape
    if C != H * W:
        raise ValueError(f"Expected C == H*W, got C={C}, H*W={H*W}")

    cost = -probs.transpose(1, 2, 0).reshape(-1, C)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

    assignment = np.empty(C, dtype=np.int64)
    assignment[row_ind] = col_ind
    return assignment


def compute_reconstruction_accuracy(im, pred, gt, num_patches: int = 4, eps: float = 0.05):
    """
    Compute proportion of correctly reconstructed patches, ignoring black patches.
    """
    C, H, W = im.shape
    assert C == 1, "[ ERROR ] Only grayscale images supported."
    assert pred.shape == gt.shape, "[ ERROR ] Prediction and ground truth must have same shape."
    assert len(pred.shape) == 1, "[ ERROR ] Predictions and ground truth must be flat vectors."

    ph = H // num_patches
    pw = W // num_patches
    assert ph == pw, "[ ERROR ] Only square patches supported."

    im_2d = np.squeeze(im)

    num_correct = 0
    num_total = 0
    i = -1

    for y in range(0, H, ph):
        for x in range(0, W, pw):
            i += 1
            if np.all(im_2d[y:y+ph, x:x+pw] < eps):
                continue
            num_total += 1
            if pred[i] == gt[i]:
                num_correct += 1

    return num_correct / num_total


def compute_total_receptive_field(net):
    """
    Compute the receptive field of the whole network.
    Uses duck typing to detect ConvLayer (has kernel_size + pad) and
    MaxPoolLayer (has size + stride but no pad).
    Returns the final receptive field size (square).
    """
    R = 1  # receptive field of input (1 pixel)
    j = 1  # jump between receptive field centers

    for layer in net.layers:
        if hasattr(layer, 'kernel_size') and hasattr(layer, 'pad'):
            # ConvLayer
            k = layer.kernel_size
            s = layer.stride
            R += (k - 1) * j
            j = j * s
        elif hasattr(layer, 'size') and hasattr(layer, 'stride') and not hasattr(layer, 'pad'):
            # MaxPoolLayer
            k = layer.size
            s = layer.stride
            R += (k - 1) * j
            j = j * s

    return R
