from tqdm import tqdm
from layers import (
    Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer, CrossEntropyLoss
)
from utils import PatchShuffleDataLoader, mount_google_drive, compute_total_receptive_field, setup_logging, save_checkpoint


def build_network():
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
    ])


if __name__ == "__main__":
    log = setup_logging()

    # # Mount Google Drive (no-op outside Colab)
    # mount_google_drive()

    # # ---- Configuration ----
    # dataset_dir = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32"
    # json_file = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32_cross_val/fold_1.json"

    ## ---- Local Configuration ----
    dataset_dir = "dataset/acv_train_32x32"
    json_file = "dataset/acv_train_32x32_cross_val/fold_1.json"

    batch_size = 4
    image_size = (32, 32)
    num_patches = 4
    num_epochs = 5
    learning_rate = 1e-3

    loader = PatchShuffleDataLoader(
        json_file,
        dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_patches=num_patches,
        shuffle=True
    )

    net = build_network()
    log.info(f"Receptive field: {compute_total_receptive_field(net)} pixels")
    log.info(f"Starting training | epochs={num_epochs} | batch_size={batch_size}")

    # ---- Training loop ----
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        iter_train = 0

        pbar = tqdm(loader.train_batches(), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for X_batch, Y_batch in pbar:
            iter_train += 1

            out_batch = net.forward(X_batch)

            loss_layer = CrossEntropyLoss()
            loss = loss_layer.forward(out_batch, Y_batch)
            epoch_loss += loss

            pbar.set_postfix(loss=f"{loss:.4f}")

            dX = loss_layer.backward()
            net.backward(dX)
            net.step(learning_rate)

        avg_loss = epoch_loss / max(iter_train, 1)
        ckpt_path = save_checkpoint(net, epoch)
        log.info(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | Checkpoint: {ckpt_path}")
