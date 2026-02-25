import os
import pickle
from tqdm import tqdm
from layers import (
    Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer, CrossEntropyLoss,
    Adam, SGD
)
from utils import (
    PatchShuffleDataLoader, mount_google_drive, compute_total_receptive_field,
    setup_logging, save_checkpoint, assign_patches, compute_reconstruction_accuracy
)


def build_network(optimizer=None):
    """Construct and return the patch-sorting CNN (Week 5 architecture)."""
    return Network([
        ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),   # 32x32 -> 16x16

        ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),   # 16x16 -> 8x8

        ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        MaxPoolLayer(size=2, stride=2),   # 8x8 -> 4x4

        ConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),

        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),

        ConvLayer(in_channels=64, out_channels=16, kernel_size=1, stride=1, pad=0),

        SoftmaxLayer(axis=1),  # softmax across channel/class dimension
    ], optimizer=optimizer)

if __name__ == "__main__":
    log = setup_logging()

    # ===============================
    # TRAINING CURRICULUM PLAN
    # ===============================
    training_plan = [

            # {
            #     "name": "stage1_2x2_2k",
            #     "num_patches": 4,
            #     "max_samples": 2000,
            #     "epochs": 120,
            #     "lr": 3e-4,
            #     "batch_size": 8,
            # },

        {
            "name": "stage2_2x2_10k",
            "num_patches": 4,
            "max_samples": 10000,
            "epochs": 80,
            "lr": 2e-4,
            "batch_size": 8,
        },

        {
            "name": "stage3_3x3_5k",
            "num_patches": 9,
            "max_samples": 5000,
            "epochs": 40,
            "lr": 1e-4,
            "batch_size": 8,
        },
    ]

    # ===============================
    # DATA PATHS
    # ===============================
    dataset_dir = "dataset/acv_train_32x32"
    json_file   = "dataset/acv_train_32x32_cross_val/fold_1.json"

    # Unique checkpoint folder per training run
    from datetime import datetime
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    previous_checkpoint = None

    # ===============================
    # RUN ALL STAGES
    # ===============================
    for stage in training_plan:

        log.info("="*60)
        log.info(f"Starting {stage['name']}")
        log.info("="*60)

        batch_size    = stage["batch_size"]
        num_patches   = stage["num_patches"]
        num_epochs    = stage["epochs"]
        learning_rate = stage["lr"]
        max_samples   = stage["max_samples"]
        image_size    = (32, 32)

        loader = PatchShuffleDataLoader(
            json_file,
            dataset_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_patches=num_patches,
            shuffle=True
        )

        if max_samples is not None:
            loader.train_files = loader.train_files[:max_samples]
            loader.val_files   = loader.val_files[:max(max_samples // 10, 100)]

        optimizer = Adam(lr=learning_rate)

        # Load previous stage or build fresh
        if previous_checkpoint is not None:
            log.info(f"Loading network from {previous_checkpoint}")
            with open(previous_checkpoint, "rb") as f:
                net = pickle.load(f)
            net.optimizer = optimizer  # swap in new optimizer (resets Adam state)
        else:
            net = build_network(optimizer=optimizer)

        log.info(
            f"Training | epochs={num_epochs} | batch={batch_size} "
            f"| samples={len(loader.train_files)} | patches={num_patches}"
        )

        # ===============================
        # TRAIN LOOP
        # ===============================
        for epoch in range(num_epochs):

            epoch_loss = 0.0
            iter_train = 0

            pbar = tqdm(
                loader.train_batches(),
                desc=f"{stage['name']} Epoch {epoch+1}/{num_epochs}",
                leave=False
            )

            for X_batch, Y_batch in pbar:
                iter_train += 1

                out_batch = net.forward(X_batch)

                loss_layer = CrossEntropyLoss()
                loss = loss_layer.forward(out_batch, Y_batch, images=X_batch)
                epoch_loss += loss

                pbar.set_postfix(loss=f"{loss:.4f}")

                dX = loss_layer.backward()
                net.backward(dX)
                net.step()

            avg_loss = epoch_loss / max(iter_train, 1)

            # ===============================
            # VALIDATION
            # ===============================
            total_acc = 0.0
            num_val = 0

            for X_val, Y_val in loader.val_batches():
                out_val = net.forward(X_val)
                B = X_val.shape[0]

                for i in range(B):
                    pred = assign_patches(out_val[i])
                    acc  = compute_reconstruction_accuracy(
                        X_val[i], pred, Y_val[i],
                        num_patches=num_patches
                    )
                    total_acc += acc
                    num_val   += 1

            val_acc = total_acc / max(num_val, 1)

            ckpt_path = save_checkpoint(net, f"{stage['name']}_epoch_{epoch}", checkpoint_dir=checkpoint_dir)
            previous_checkpoint = ckpt_path

            log.info(
                f"{stage['name']} | Epoch {epoch+1}/{num_epochs} "
                f"| Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} "
                f"| Checkpoint: {ckpt_path}"
            )

    log.info("ALL TRAINING STAGES COMPLETE")

    
# if __name__ == "__main__":
#     log = setup_logging()

#     # # Mount Google Drive (no-op outside Colab)
#     # mount_google_drive()

#     # # ---- Configuration ----
#     # dataset_dir = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32"
#     # json_file = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32_cross_val/fold_1.json"

#     ## ---- Local Configuration ----
#     dataset_dir = "dataset/acv_train_32x32"
#     json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"

#     # M1 Mac settings — CPU-only numpy, no GPU
#     # Use max_samples to train on a subset locally (None = full dataset)
#     batch_size    = 8
#     image_size    = (32, 32)
#     num_patches   = 4
#     num_epochs    = 30
#     learning_rate = 3e-4
#     max_samples   = 2000
#     loader = PatchShuffleDataLoader(
#         json_file,
#         dataset_dir,
#         batch_size=batch_size,
#         image_size=image_size,
#         num_patches=num_patches,
#         shuffle=True
#     )

#     # Subset for faster local training
#     if max_samples is not None:
#         loader.train_files = loader.train_files[:max_samples]
#         loader.val_files = loader.val_files[:max(max_samples // 10, 100)]

#     optimizer = Adam(lr=learning_rate)
#     net = build_network(optimizer=optimizer)
#     log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
#     log.info(f"Starting training | epochs={num_epochs} | batch_size={batch_size} | samples={len(loader.train_files)}")

#     # ---- Training loop ----
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         iter_train = 0

#         pbar = tqdm(loader.train_batches(), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
#         for X_batch, Y_batch in pbar:
#             iter_train += 1

#             out_batch = net.forward(X_batch)

#             loss_layer = CrossEntropyLoss()
#             loss = loss_layer.forward(out_batch, Y_batch, images=X_batch)
#             epoch_loss += loss

#             pbar.set_postfix(loss=f"{loss:.4f}")

#             dX = loss_layer.backward()
#             net.backward(dX)
#             net.step()

#         avg_loss = epoch_loss / max(iter_train, 1)

#         # ---- Validation accuracy ----
#         total_acc = 0.0
#         num_val = 0

#         for X_val, Y_val in loader.val_batches():
#             out_val = net.forward(X_val)
#             B = X_val.shape[0]

#             for i in range(B):
#                 pred = assign_patches(out_val[i])
#                 acc = compute_reconstruction_accuracy(X_val[i], pred, Y_val[i], num_patches=num_patches)
#                 total_acc += acc
#                 num_val += 1

#         val_acc = total_acc / max(num_val, 1)

#         ckpt_path = save_checkpoint(net, epoch)
#         log.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Checkpoint: {ckpt_path}")
