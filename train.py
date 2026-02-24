from layers import (
    Network, ConvLayer, ReLULayer, MaxPoolLayer, SoftmaxLayer, CrossEntropyLoss
)
from utils import PatchShuffleDataLoader, mount_google_drive, compute_total_receptive_field


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
    # Mount Google Drive (no-op outside Colab)
    mount_google_drive()

    # ---- Configuration ----
    dataset_dir = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32"
    json_file = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32_cross_val/fold_1.json"

    batch_size = 32
    image_size = (32, 32)
    num_patches = 4
    num_epochs = 50
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
    print(f"[INFO] Receptive field: {compute_total_receptive_field(net)} pixels")

    # ---- Training loop ----
    for epoch in range(num_epochs):
        iter_train = 0

        for X_batch, Y_batch in loader.train_batches():
            iter_train += 1

            out_batch = net.forward(X_batch)   # (B, C, H, W)

            loss_layer = CrossEntropyLoss()
            loss = loss_layer.forward(out_batch, Y_batch)

            print(f"[INFO] Mode: Training, Epoch: {epoch}, "
                  f"Iteration: {iter_train}, Loss: {loss}")

            dX = loss_layer.backward()
            net.backward(dX)
            net.step(learning_rate)
