import argparse
import os
import pickle
import numpy as np
import cv2 as cv
import scipy.optimize

# Required for pickle to reconstruct the network object
from layers import (
    Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer,
    SGD, Adam
)


def assign_patches(probs):
    """
    Solve the patch assignment problem using predicted probabilities.

    Args:
        probs: np.ndarray of shape (C, H, W), where C = H * W,
               probs[i, h, w] = probability that patch (h, w)
               was originally at position i.

    Returns:
        np.ndarray of shape (C,) where assignment[i] is the original index
        of the patch currently at shuffled position i.
    """
    C, H, W = probs.shape
    assert C == H * W, "Invalid shape for probabilities."

    cost = -probs.transpose(1, 2, 0).reshape(-1, C)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

    assignment = np.zeros(C, dtype=int)
    assignment[row_ind] = col_ind
    return assignment


def load_image(filepath):
    """Load a 32x32 shuffled image and return as (1, 1, 32, 32) float32 array."""
    image = cv.imread(filepath)
    if image is None:
        raise ValueError(f"Failed to load image: {filepath}")

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (32, 32))
    image = image.astype(np.float32) / 255.0
    image = image[np.newaxis, np.newaxis, :, :]  # (1, 1, 32, 32)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True, help="Directory containing shuffled test images")
    parser.add_argument("--model", required=True, help="Path to model weights (pickle file)")
    args = parser.parse_args()

    # Load model
    with open(args.model, "rb") as f:
        net = pickle.load(f)

    # Get all PNG images in the test directory
    image_files = sorted([
        f for f in os.listdir(args.test_dir)
        if f.lower().endswith(".png")
    ])

    default_assignment = list(range(16))  # identity permutation fallback

    for image_name in image_files:
        filepath = os.path.join(args.test_dir, image_name)

        try:
            image = load_image(filepath)
            probs = net.forward(image)  # (1, 16, 4, 4)
            probs = probs[0]            # (16, 4, 4)
            assignment = assign_patches(probs)
        except Exception:
            assignment = default_assignment

        preds_str = ",".join(str(p) for p in assignment)
        print(f"{image_name},{preds_str}")


if __name__ == "__main__":
    main()
