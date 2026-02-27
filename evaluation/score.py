"""
Score predictions against ground truth — simulates the grader.

Usage:
    python test.py --test-dir evaluation/mock_test --model final.pkl > evaluation/predictions.csv
    python evaluation/score.py --predictions evaluation/predictions.csv --ground-truth evaluation/mock_test_gt.csv --test-dir evaluation/mock_test
"""
import argparse
import os
import numpy as np
import cv2 as cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Your test.py output (CSV)")
    parser.add_argument("--ground-truth", required=True, help="Ground truth CSV")
    parser.add_argument("--test-dir", required=True, help="Directory with shuffled images")
    args = parser.parse_args()

    # Parse ground truth
    gt = {}
    with open(args.ground_truth) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            gt[parts[0]] = np.array([int(x) for x in parts[1:]])

    # Parse predictions
    preds = {}
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            preds[parts[0]] = np.array([int(x) for x in parts[1:]])

    # Score
    eps = 0.05
    N = 4
    total_acc = 0.0
    num_images = 0
    missing = 0

    for fname, gt_label in gt.items():
        if fname not in preds:
            missing += 1
            continue

        pred_label = preds[fname]

        # Load shuffled image to find black patches
        img = cv.imread(os.path.join(args.test_dir, fname), cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_float = img.astype(np.float32) / 255.0

        H, W = img_float.shape
        ph, pw = H // N, W // N

        correct = 0
        total = 0
        idx = -1
        for y in range(0, H, ph):
            for x in range(0, W, pw):
                idx += 1
                # Skip all-black patches
                if np.all(img_float[y:y+ph, x:x+pw] < eps):
                    continue
                total += 1
                if pred_label[idx] == gt_label[idx]:
                    correct += 1

        acc = correct / max(total, 1)
        total_acc += acc
        num_images += 1

    avg_acc = total_acc / max(num_images, 1)

    # Determine mark
    if avg_acc < 0.30:
        mark = 0
    elif avg_acc < 0.40:
        mark = 35
    elif avg_acc < 0.50:
        mark = 42
    elif avg_acc < 0.60:
        mark = 50
    else:
        mark = 70

    print(f"{'='*50}")
    print(f"GRADING RESULTS")
    print(f"{'='*50}")
    print(f"Images in ground truth: {len(gt)}")
    print(f"Images in predictions:  {len(preds)}")
    print(f"Missing predictions:    {missing}")
    print(f"Images scored:          {num_images}")
    print(f"Average accuracy:       {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"{'='*50}")
    print(f"MARK: {mark}/70")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
