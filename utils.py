import random
import json
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.optimize
from typing import Tuple, List

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

        with open(json_file, 'r') as f:
            data = json.load(f)
        self.train_files = data['train']
        self.val_files = data['val']

    def _shuffle_patches(self, image: np.ndarray, indices: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide image into patches, shuffle them, and return the shuffled image
        along with the permutation labels.

        Args:
            image (np.ndarray): Input image of shape (H, W, C)
            indices (np.ndarray, optional): Predefined permutation of patch indices.
                                            If None, a random shuffle is used.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Shuffled image and permutation indices
        """
        H, W = image.shape[0], image.shape[1]
        N = self.num_patches
        ph, pw = H // N, W // N

        if H % N != 0 or W % N != 0:
            raise ValueError(
                f"Image size ({H}, {W}) not divisible by num_patches={N}. "
                "Choose a different num_patches or resize/crop the image."
            )

        self.patch_size = (ph, pw)

        patches = []
        for i in range(N):
            for j in range(N):
                patch = image[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
                patches.append(patch)
        patches = np.array(patches)  # (N^2, ph, pw, C)

        num_total = N * N
        if indices is None:
            indices = np.random.permutation(num_total)
        else:
            indices = np.asarray(indices)
            if indices.shape != (num_total,):
                raise ValueError(f"indices must have shape ({num_total},), got {indices.shape}")
            if set(indices.tolist()) != set(range(num_total)):
                raise ValueError("indices must be a permutation of 0..N^2-1 (no repeats, none missing)")

        shuffled_image = np.empty_like(image)
        label = np.empty(num_total, dtype=np.int64)

        for new_pos, original_id in enumerate(indices):
            out_i = new_pos // N
            out_j = new_pos % N
            shuffled_image[out_i*ph:(out_i+1)*ph, out_j*pw:(out_j+1)*pw, :] = patches[original_id]
            label[original_id] = new_pos

        return shuffled_image, label

    def _load_image(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess a single image with patch shuffling (CHW + permutation labels).
        """
        full_path = os.path.join(self.dataset_dir, filepath)
        image = cv.imread(full_path)

        if image is None:
            raise ValueError(f"Failed to load image: {full_path}")

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
        if getattr(self, "shuffle", False):
            random.shuffle(files)

        for start in range(0, len(files), self.batch_size):
            batch_files = files[start:start + self.batch_size]

            X, Y = [], []
            for fname in batch_files:
                img, label = self._load_image(fname)
                X.append(img)
                Y.append(label)

            X = np.stack(X, axis=0)  # (B, C, H, W)
            Y = np.stack(Y, axis=0)  # (B, N^2)

            yield X, Y

    def train_batches(self):
        return self._get_batches(self.train_files)

    def val_batches(self):
        return self._get_batches(self.val_files)

    def display_training_image(self, image: np.ndarray):
        new_im = np.transpose(image, (1, 2, 0))
        new_im = np.clip(new_im * 255.0, 0, 255).astype(np.uint8)
        plt.imshow(new_im)
        plt.axis("off")
        plt.show()

    def display_reconstructed_training_image(self, image: np.ndarray, label: np.ndarray):
        new_im = np.transpose(image, (1, 2, 0))
        new_im = np.clip(new_im * 255.0, 0, 255).astype(np.uint8)
        new_im, _ = self._shuffle_patches(new_im, label)
        plt.imshow(new_im)
        plt.axis("off")
        plt.show()


def assign_patches(probs):
    """
    Solve the patch assignment problem using predicted probabilities.

    Args:
        probs: np.ndarray of shape (C, H, W),
               where C = H * W,
               probs[i, h, w] = probability that patch (h, w)
               was originally at position i.

    Returns:
        np.ndarray of shape (C,) containing assigned original position index for each patch.
    """
    C, H, W = probs.shape
    assert C == H * W, "[ ERROR ] Invalid shape for probabilities."

    cost = -probs.transpose(1, 2, 0).reshape(-1, C)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

    assignment = np.zeros(C, dtype=int)
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
