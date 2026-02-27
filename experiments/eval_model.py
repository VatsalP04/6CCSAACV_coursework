import argparse
import json
import os
import sys
import random
import subprocess
import numpy as np
import cv2 as cv

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


def shuffle_patches(image, num_patches=4):
    """
    Shuffle patches of a grayscale image and return shuffled image + ground truth.

    Args:
        image: (H, W) grayscale image, uint8
        num_patches: patches per side (4 = 4x4 grid)

    Returns:
        shuffled: (H, W) shuffled image
        label: (N^2,) where label[original_id] = new_position
    """
    H, W = image.shape
    N = num_patches
    ph, pw = H // N, W // N

    patches = []
    for i in range(N):
        for j in range(N):
            patches.append(image[i*ph:(i+1)*ph, j*pw:(j+1)*pw])

    num_total = N * N
    indices = np.random.permutation(num_total)

    shuffled = np.empty_like(image)
    label = np.empty(num_total, dtype=int)

    for new_pos, original_id in enumerate(indices):
        out_i = new_pos // N
        out_j = new_pos % N
        shuffled[out_i*ph:(out_i+1)*ph, out_j*pw:(out_j+1)*pw] = patches[original_id]
        label[original_id] = new_pos

    return shuffled, label


def compute_accuracy(pred, gt, shuffled_image, num_patches=4, eps=0.05):
    """
    Compute patch accuracy excluding black patches (matching professor's metric).
    """
    H, W = shuffled_image.shape
    N = num_patches
    ph, pw = H // N, W // N
    im_float = shuffled_image.astype(np.float32) / 255.0

    num_correct = 0
    num_total = 0
    idx = -1

    for y in range(0, H, ph):
        for x in range(0, W, pw):
            idx += 1
            if np.all(im_float[y:y+ph, x:x+pw] < eps):
                continue
            num_total += 1
            if pred[idx] == gt[idx]:
                num_correct += 1

    return num_correct / max(num_total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model pickle file")
    parser.add_argument("--dataset-dir", default="dataset/acv_train_32x32",
                        help="Directory containing original images")
    parser.add_argument("--json-file", default="dataset/acv_train_32x32_cross_val/fold_1.json",
                        help="Fold JSON file (uses val split)")
    parser.add_argument("--num-images", type=int, default=200,
                        help="Number of val images to evaluate on")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup temp directories (clear previous run)
    temp_dir = "evaluation"
    shuffled_dir = os.path.join(temp_dir, "shuffled")
    if os.path.exists(shuffled_dir):
        for f in os.listdir(shuffled_dir):
            os.remove(os.path.join(shuffled_dir, f))
    os.makedirs(shuffled_dir, exist_ok=True)

    # Get random subset from validation split
    with open(args.json_file, "r") as f:
        fold_data = json.load(f)
    val_images = fold_data["val"]
    random.shuffle(val_images)
    selected = val_images[:args.num_images]

    print(f"Generating {len(selected)} shuffled images...")

    # Shuffle patches and save
    gt_path = os.path.join(temp_dir, "ground_truth.csv")
    ground_truth = {}

    with open(gt_path, "w") as gt_file:
        for fname in selected:
            img = cv.imread(os.path.join(args.dataset_dir, fname))
            if img is None:
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (32, 32))

            shuffled, label = shuffle_patches(img, num_patches=4)
            ground_truth[fname] = label

            out_path = os.path.join(shuffled_dir, fname)
            cv.imwrite(out_path, shuffled)

            label_str = ",".join(str(p) for p in label)
            gt_file.write(f"{fname},{label_str}\n")

    print(f"Saved shuffled images to {shuffled_dir}")
    print(f"Saved ground truth to {gt_path}")

    # Run test.py
    pred_path = os.path.join(temp_dir, "predictions.csv")
    cmd = f"python test.py --test-dir {shuffled_dir} --model {args.model}"
    print(f"\nRunning: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"test.py failed:\n{result.stderr}")
        return

    with open(pred_path, "w") as f:
        f.write(result.stdout)

    # Parse predictions
    predictions = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split(",")
        name = parts[0]
        preds = np.array([int(p) for p in parts[1:]])
        predictions[name] = preds

    # Compare
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")

    default_pred = list(range(16))
    total_acc = 0.0
    num_evaluated = 0
    num_default = 0

    for fname, gt_label in ground_truth.items():
        if fname not in predictions:
            print(f"  WARNING: {fname} missing from predictions!")
            continue

        pred_label = predictions[fname]

        if list(pred_label) == default_pred:
            num_default += 1

        img = cv.imread(os.path.join(shuffled_dir, fname), cv.IMREAD_GRAYSCALE)
        acc = compute_accuracy(pred_label, gt_label, img, num_patches=4)
        total_acc += acc
        num_evaluated += 1

    avg_acc = total_acc / max(num_evaluated, 1)

    print(f"Images evaluated: {num_evaluated}")
    print(f"Default predictions (failed to load): {num_default}")
    print(f"Average accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
