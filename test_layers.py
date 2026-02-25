import numpy as np
from layers import (
    im2col, col2im, ConvLayer, ReLULayer, MaxPoolLayer,
    SoftmaxLayer, CrossEntropyLoss, Network, SGD, Adam
)

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1


# ===========================
# 1. im2col
# ===========================
def test_im2col():
    print("\n[im2col]")
    np.random.seed(0)
    X = np.random.randn(2, 3, 4, 4).astype(np.float32)

    H_out, W_out, cols = im2col(X, kernel_size=3, stride=1, pad=1)
    check("output height", H_out == 4)
    check("output width", W_out == 4)
    check("cols shape", cols.shape == (3 * 3 * 3, 2 * 4 * 4))

    # No padding case
    H_out2, W_out2, cols2 = im2col(X, kernel_size=3, stride=1, pad=0)
    check("no-pad output height", H_out2 == 2)
    check("no-pad output width", W_out2 == 2)
    check("no-pad cols shape", cols2.shape == (3 * 3 * 3, 2 * 2 * 2))

    # Stride 2
    H_out3, W_out3, cols3 = im2col(X, kernel_size=2, stride=2, pad=0)
    check("stride2 output height", H_out3 == 2)
    check("stride2 output width", W_out3 == 2)


# ===========================
# 2. col2im
# ===========================
def test_col2im():
    print("\n[col2im]")
    np.random.seed(1)
    X = np.random.randn(1, 1, 4, 4).astype(np.float32)

    # No padding, stride 1, kernel 1 => perfect round-trip
    H_out, W_out, cols = im2col(X, kernel_size=1, stride=1, pad=0)
    X_rec = col2im(cols, X.shape, kernel_size=1, stride=1, pad=0)
    check("round-trip k=1 s=1 p=0", np.allclose(X, X_rec, atol=1e-6))

    # With padding: col2im should reconstruct shape
    H_out2, W_out2, cols2 = im2col(X, kernel_size=3, stride=1, pad=1)
    X_rec2 = col2im(cols2, X.shape, kernel_size=3, stride=1, pad=1)
    check("col2im output shape with pad", X_rec2.shape == X.shape)


# ===========================
# 3. ConvLayer
# ===========================
def test_conv_layer():
    print("\n[ConvLayer]")
    np.random.seed(2)
    conv = ConvLayer(in_channels=1, out_channels=4, kernel_size=3, stride=1, pad=1)
    X = np.random.randn(2, 1, 8, 8).astype(np.float32)

    out = conv.forward(X)
    check("forward shape", out.shape == (2, 4, 8, 8))

    # Check specific output value for reproducibility
    np.random.seed(2)
    conv2 = ConvLayer(in_channels=1, out_channels=4, kernel_size=3, stride=1, pad=1)
    out2 = conv2.forward(X)
    check("deterministic forward", np.allclose(out, out2, atol=1e-6))

    # Backward
    dY = np.random.randn(*out.shape).astype(np.float32)
    dX = conv.backward(dY)
    check("backward dX shape", dX.shape == X.shape)
    check("dW computed", hasattr(conv, 'dW') and conv.dW.shape == conv.W.shape)
    check("db computed", hasattr(conv, 'db') and conv.db.shape == conv.b.shape)

    # Stride 2
    np.random.seed(3)
    conv3 = ConvLayer(in_channels=1, out_channels=2, kernel_size=3, stride=2, pad=1)
    out3 = conv3.forward(X)
    check("stride=2 forward shape", out3.shape == (2, 2, 4, 4))


# ===========================
# 4. ReLULayer
# ===========================
def test_relu_layer():
    print("\n[ReLULayer]")
    relu = ReLULayer()
    X = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]).reshape(1, 1, 1, 5)

    out = relu.forward(X)
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]]).reshape(1, 1, 1, 5)
    check("forward values", np.allclose(out, expected))

    dY = np.ones_like(out)
    dX = relu.backward(dY)
    expected_grad = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]]).reshape(1, 1, 1, 5)
    check("backward gradient mask", np.allclose(dX, expected_grad))


# ===========================
# 5. MaxPoolLayer
# ===========================
def test_maxpool_layer():
    print("\n[MaxPoolLayer]")
    pool = MaxPoolLayer(size=2, stride=2)
    X = np.array([[
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    ]], dtype=np.float32)  # (1, 1, 4, 4)

    out = pool.forward(X)
    check("forward shape", out.shape == (1, 1, 2, 2))
    expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
    check("forward max values", np.allclose(out, expected))

    dY = np.ones((1, 1, 2, 2), dtype=np.float32)
    dX = pool.backward(dY)
    check("backward shape", dX.shape == (1, 1, 4, 4))
    # Gradient should only be at max positions
    check("backward max routing", dX[0, 0, 1, 1] == 1.0)  # position of 6
    check("backward zero elsewhere", dX[0, 0, 0, 0] == 0.0)  # position of 1

    # Multi-channel
    np.random.seed(4)
    pool2 = MaxPoolLayer(size=2, stride=2)
    X2 = np.random.randn(2, 3, 8, 8).astype(np.float32)
    out2 = pool2.forward(X2)
    check("multi-channel shape", out2.shape == (2, 3, 4, 4))


# ===========================
# 6. SoftmaxLayer
# ===========================
def test_softmax_layer():
    print("\n[SoftmaxLayer]")
    sm = SoftmaxLayer(axis=1)
    np.random.seed(5)
    X = np.random.randn(2, 4, 3, 3).astype(np.float32)

    out = sm.forward(X)
    check("forward shape", out.shape == X.shape)
    check("values in [0,1]", np.all(out >= 0) and np.all(out <= 1))

    sums = np.sum(out, axis=1)
    check("sums to 1 along axis", np.allclose(sums, 1.0, atol=1e-6))

    # Backward
    dY = np.random.randn(*X.shape).astype(np.float32)
    dX = sm.backward(dY)
    check("backward shape", dX.shape == X.shape)

    # Numerical gradient check (spot check one element)
    eps = 1e-5
    i, j, k, l = 0, 1, 1, 1
    X_plus = X.copy()
    X_plus[i, j, k, l] += eps
    X_minus = X.copy()
    X_minus[i, j, k, l] -= eps
    sm_plus = SoftmaxLayer(axis=1)
    sm_minus = SoftmaxLayer(axis=1)
    out_plus = sm_plus.forward(X_plus)
    out_minus = sm_minus.forward(X_minus)
    numerical_grad = np.sum(dY * (out_plus - out_minus)) / (2 * eps)
    check("numerical gradient", abs(dX[i, j, k, l] - numerical_grad) < 5e-3)


# ===========================
# 7. CrossEntropyLoss
# ===========================
def test_cross_entropy_loss():
    print("\n[CrossEntropyLoss]")
    np.random.seed(6)

    B, C, H, W = 2, 4, 2, 2
    # Create valid softmax output
    sm = SoftmaxLayer(axis=1)
    logits = np.random.randn(B, C, H, W).astype(np.float32)
    probs = sm.forward(logits)

    # Random permutation labels
    labels = np.zeros((B, C), dtype=np.int64)
    for b in range(B):
        labels[b] = np.random.permutation(C)

    loss_fn = CrossEntropyLoss()
    loss = loss_fn.forward(probs, labels)
    check("loss is scalar", isinstance(loss, (float, np.floating)))
    check("loss is positive", loss > 0)

    dX = loss_fn.backward()
    check("backward shape", dX.shape == probs.shape)

    # Test with black patch masking
    images = np.random.randn(B, 1, 8, 8).astype(np.float32)
    images[:, :, 0:4, 0:4] = 0.01  # make top-left patch black

    loss_fn2 = CrossEntropyLoss()
    loss_masked = loss_fn2.forward(probs, labels, images=images, eps_black=0.05)
    check("masked loss is positive", loss_masked > 0)

    dX_masked = loss_fn2.backward()
    check("masked backward shape", dX_masked.shape == probs.shape)


# ===========================
# 8. SGD
# ===========================
def test_sgd():
    print("\n[SGD]")
    np.random.seed(7)
    conv = ConvLayer(in_channels=1, out_channels=2, kernel_size=3, stride=1, pad=1)
    X = np.random.randn(1, 1, 4, 4).astype(np.float32)

    out = conv.forward(X)
    dY = np.random.randn(*out.shape).astype(np.float32)
    conv.backward(dY)

    W_before = conv.W.copy()
    b_before = conv.b.copy()

    sgd = SGD(lr=0.01)
    sgd.step([conv])

    check("weights updated", not np.allclose(conv.W, W_before))
    check("bias updated", not np.allclose(conv.b, b_before))
    check("W = W - lr*dW", np.allclose(conv.W, W_before - 0.01 * conv.dW, atol=1e-7))
    check("b = b - lr*db", np.allclose(conv.b, b_before - 0.01 * conv.db, atol=1e-7))


# ===========================
# 9. Adam
# ===========================
def test_adam():
    print("\n[Adam]")
    np.random.seed(8)
    conv = ConvLayer(in_channels=1, out_channels=2, kernel_size=3, stride=1, pad=1)
    X = np.random.randn(1, 1, 4, 4).astype(np.float32)

    out = conv.forward(X)
    dY = np.random.randn(*out.shape).astype(np.float32)
    conv.backward(dY)

    W_before = conv.W.copy()

    adam = Adam(lr=0.001)
    adam.step([conv])

    check("weights updated", not np.allclose(conv.W, W_before))
    check("momentum state created", hasattr(conv, 'mW'))
    check("velocity state created", hasattr(conv, 'vW'))
    check("timestep incremented", adam.t == 1)

    # Second step
    out2 = conv.forward(X)
    dY2 = np.random.randn(*out2.shape).astype(np.float32)
    conv.backward(dY2)
    W_after_1 = conv.W.copy()
    adam.step([conv])
    check("second step updates", not np.allclose(conv.W, W_after_1))
    check("timestep is 2", adam.t == 2)


# ===========================
# 10. Network full forward
# ===========================
def test_network():
    print("\n[Network full forward]")
    np.random.seed(9)
    # Build network directly to avoid importing train.py (which needs cv2)
    net = Network([
        ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        MaxPoolLayer(size=2, stride=2),
        ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        MaxPoolLayer(size=2, stride=2),
        ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        MaxPoolLayer(size=2, stride=2),
        ConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, pad=1),
        ReLULayer(),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
        ConvLayer(in_channels=64, out_channels=16, kernel_size=1, stride=1, pad=0),
        SoftmaxLayer(axis=1),
    ], optimizer=Adam(lr=1e-3))

    X = np.random.randn(1, 1, 32, 32).astype(np.float32)
    out = net.forward(X)

    check("output shape", out.shape == (1, 16, 4, 4))
    check("softmax range [0,1]", np.all(out >= 0) and np.all(out <= 1))

    # Softmax sums to 1 along channel axis
    sums = np.sum(out, axis=1)
    check("softmax sums to 1", np.allclose(sums, 1.0, atol=1e-5))

    # Backward should not error
    dY = np.random.randn(*out.shape).astype(np.float32)
    dX = net.backward(dY)
    check("backward output shape", dX.shape == X.shape)


# ===========================
# RUN ALL
# ===========================
if __name__ == "__main__":
    test_im2col()
    test_col2im()
    test_conv_layer()
    test_relu_layer()
    test_maxpool_layer()
    test_softmax_layer()
    test_cross_entropy_loss()
    test_sgd()
    test_adam()
    test_network()

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*50}")

    if failed > 0:
        exit(1)
