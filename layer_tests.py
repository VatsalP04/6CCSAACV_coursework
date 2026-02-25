import numpy as np
import torch
from layers import ConvLayer, ReLULayer, MaxPoolLayer, CrossEntropyLoss, Network


def test_conv_layer():
    B, C, H, W = 2, 3, 5, 5
    F = 4
    kernel_size = 3
    stride = 1
    pad = 1

    X_np = np.random.randn(B, C, H, W).astype(np.float32)
    X_torch = torch.tensor(X_np, requires_grad=True)

    conv_np = ConvLayer(C, F, kernel_size=kernel_size, stride=stride, pad=pad)

    conv_torch = torch.nn.Conv2d(
        in_channels=C, out_channels=F,
        kernel_size=kernel_size, stride=stride, padding=pad, bias=True
    )
    with torch.no_grad():
        conv_torch.weight.copy_(torch.tensor(conv_np.W))
        conv_torch.bias.copy_(torch.tensor(conv_np.b))

    # Forward
    out_np = conv_np.forward(X_np)
    out_torch = conv_torch(X_torch)

    assert np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5), \
        "[ERROR] Convolutional layer: Forward outputs do not match PyTorch."

    # Backward
    dY_np = np.random.randn(*out_np.shape).astype(np.float32)
    dY_torch = torch.tensor(dY_np)

    dX_np = conv_np.backward(dY_np)
    out_torch.backward(dY_torch)
    dX_torch = X_torch.grad.detach().numpy()

    assert np.allclose(conv_np.dW, conv_torch.weight.grad.detach().numpy(), atol=1e-5), \
        "[ERROR] Convolutional layer: dW mismatch."
    assert np.allclose(conv_np.db, conv_torch.bias.grad.detach().numpy(), atol=1e-5), \
        "[ERROR] Convolutional layer: db mismatch."
    assert np.allclose(dX_np, dX_torch, atol=1e-5), \
        "[ERROR] Convolutional layer: dX mismatch."

    print("[INFO] Convolution layer tests passed!")


def test_relu_layer():
    B, C, H, W = 2, 3, 8, 8

    X_np = np.random.randn(B, C, H, W).astype(np.float32)
    X_torch = torch.tensor(X_np, requires_grad=True)

    relu_np = ReLULayer()
    relu_torch = torch.nn.ReLU()

    # Forward
    out_np = relu_np.forward(X_np)
    out_torch = relu_torch(X_torch)

    assert np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5), \
        "[ ERROR ] ReLU layer: Forward outputs do not match PyTorch."

    # Backward
    dY_np = np.random.randn(*out_np.shape).astype(np.float32)
    dY_torch = torch.tensor(dY_np)

    dX_np = relu_np.backward(dY_np)
    out_torch.backward(dY_torch)
    dX_torch = X_torch.grad.detach().numpy()

    assert np.allclose(dX_np, dX_torch, atol=1e-5), \
        "[ ERROR ] ReLU layer: dX mismatch."

    print("[ INFO ] ReLU layer tests passed!")


def test_maxpool_layer():
    B, C, H, W = 2, 3, 8, 8
    kernel_size = 2
    stride = 2

    X_np = np.random.randn(B, C, H, W).astype(np.float32)
    X_torch = torch.tensor(X_np, requires_grad=True)

    maxpool_np = MaxPoolLayer(size=kernel_size, stride=stride)
    maxpool_torch = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    # Forward
    out_np = maxpool_np.forward(X_np)
    out_torch = maxpool_torch(X_torch)

    assert np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5), \
        "[ ERROR ] MaxPool layer: Forward outputs do not match PyTorch."

    # Backward
    dY_np = np.random.randn(*out_np.shape).astype(np.float32)
    dY_torch = torch.tensor(dY_np)

    dX_np = maxpool_np.backward(dY_np)
    out_torch.backward(dY_torch)
    dX_torch = X_torch.grad.detach().numpy()

    assert np.allclose(dX_np, dX_torch, atol=1e-5), \
        "[ ERROR ] MaxPool layer: dX mismatch."

    print("[ INFO ] MaxPool layer tests passed!")


def test_crossentropy_loss_layer():
    B, C, H, W = 2, 4, 3, 3

    logits = np.random.randn(B * H * W, C).astype(np.float32)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    logprobs = np.log(probs)

    targets = np.random.randint(0, C, size=(B * H * W,))

    logprobs_torch = torch.tensor(logprobs, requires_grad=True)
    targets_torch = torch.tensor(targets, dtype=torch.long)

    probs_np = probs.reshape(B * H * W, 1, 1, C).transpose(0, 3, 1, 2)
    targets_np = targets.reshape(B * H * W, 1)

    # PyTorch NLLLoss
    ce_torch = torch.nn.NLLLoss()
    loss_torch = ce_torch(logprobs_torch, targets_torch)

    # Custom CrossEntropyLoss
    ce_np = CrossEntropyLoss()
    loss_np = ce_np.forward(probs_np, targets_np)

    assert np.allclose(loss_np, loss_torch.item(), atol=1e-5), \
        "[ ERROR ] CrossEntropyLoss: Forward outputs do not match PyTorch."

    # Backward
    loss_torch.backward()
    dX_torch_probs = logprobs_torch.grad.detach().numpy() / (probs + 1e-12)

    dX_np = ce_np.backward().transpose(0, 2, 3, 1).reshape(B * H * W, C)

    assert np.allclose(dX_np, dX_torch_probs, atol=1e-5), \
        "[ ERROR ] CrossEntropyLoss: Backward gradients mismatch."

    print("[ INFO ] CrossEntropyLoss layer tests passed!")


if __name__ == "__main__":
    test_conv_layer()
    test_relu_layer()
    test_maxpool_layer()
    test_crossentropy_loss_layer()
