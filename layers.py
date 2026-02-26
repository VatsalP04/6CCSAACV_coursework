import numpy as np
from numpy.lib.stride_tricks import as_strided

# CuPy is used for GPU-accelerated matrix ops in ConvLayer.
# Falls back to NumPy transparently on CPU-only machines (e.g. Mac laptops).
try:
    import cupy as xp
except ImportError:
    import numpy as xp


def im2col(X, kernel_size, stride=1, pad=0):
    """
    Convert image patches to columns using stride tricks for efficient convolution.

    Args:
        X: ndarray of shape (B, C, H, W)
        kernel_size: int or (KH, KW)
        stride: int
        pad: int

    Returns:
        H_out: int
        W_out: int
        cols: ndarray of shape (C*KH*KW, B*H_out*W_out)
    """
    B, C, H, W = X.shape

    if isinstance(kernel_size, int):
        KH, KW = kernel_size, kernel_size
    else:
        KH, KW = kernel_size

    if pad > 0:
        X_p = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                      mode="constant", constant_values=0)
    else:
        X_p = X

    # Contiguous memory required for stride tricks
    X_p = np.ascontiguousarray(X_p)
    H_p, W_p = X_p.shape[2], X_p.shape[3]

    H_out = (H_p - KH) // stride + 1
    W_out = (W_p - KW) // stride + 1

    # Extract all patches as a 6D view without copying data
    sB, sC, sH, sW = X_p.strides
    patches = as_strided(
        X_p,
        shape=(B, C, H_out, W_out, KH, KW),
        strides=(sB, sC, sH * stride, sW * stride, sH, sW),
        writeable=False
    )

    # Reshape to (C*KH*KW, B*H_out*W_out) for matrix multiply with filters
    cols = patches.transpose(1, 4, 5, 0, 2, 3).reshape(C * KH * KW, B * H_out * W_out)

    return H_out, W_out, cols


def _col2im_addat(cols, input_shape, kernel_size, stride, pad):
    """General col2im using np.add.at scatter-add. Works for any stride."""
    B, C, H, W = input_shape

    if isinstance(kernel_size, int):
        KH, KW = kernel_size, kernel_size
    else:
        KH, KW = kernel_size

    H_p = H + 2 * pad
    W_p = W + 2 * pad
    H_out = (H_p - KH) // stride + 1
    W_out = (W_p - KW) // stride + 1

    dX_p = np.zeros((B, C, H_p, W_p), dtype=cols.dtype)

    # Reshape cols back to (B, C, H_out, W_out, KH, KW)
    vals = cols.reshape(C, KH, KW, B, H_out, W_out).transpose(3, 0, 4, 5, 1, 2)

    # Build scatter indices for each output position + kernel offset
    h0 = np.arange(H_out) * stride
    w0 = np.arange(W_out) * stride
    kh = np.arange(KH)
    kw = np.arange(KW)

    H_idx = (h0[:, None] + kh[None, :])[:, None, :, None]
    W_idx = (w0[:, None] + kw[None, :])[None, :, None, :]
    B_idx = np.arange(B)[:, None, None, None, None, None]
    C_idx = np.arange(C)[None, :, None, None, None, None]

    # Broadcast all indices to (B, C, H_out, W_out, KH, KW)
    shape_6d = (B, C, H_out, W_out, KH, KW)
    H_idx = np.broadcast_to(H_idx[None, None, :, :, :, :], shape_6d)
    W_idx = np.broadcast_to(W_idx[None, None, :, :, :, :], shape_6d)
    B_idx = np.broadcast_to(B_idx, shape_6d)
    C_idx = np.broadcast_to(C_idx, shape_6d)

    np.add.at(dX_p, (B_idx, C_idx, H_idx, W_idx), vals)

    if pad > 0:
        return dX_p[:, :, pad:pad + H, pad:pad + W]
    return dX_p


def _col2im_stride1(cols, input_shape, kernel_size, pad):
    """Fast col2im for stride=1 using slice accumulation instead of scatter."""
    B, C, H, W = input_shape

    if isinstance(kernel_size, int):
        KH, KW = kernel_size, kernel_size
    else:
        KH, KW = kernel_size

    H_p = H + 2 * pad
    W_p = W + 2 * pad
    H_out = (H_p - KH) + 1
    W_out = (W_p - KW) + 1

    dX_p = np.zeros((B, C, H_p, W_p), dtype=cols.dtype)

    # Reshape to (B, C, KH, KW, H_out, W_out) for per-offset accumulation
    cols6 = cols.reshape(C, KH, KW, B, H_out, W_out).transpose(3, 0, 1, 2, 4, 5)

    # Accumulate each kernel offset via slicing (no scatter needed for stride=1)
    for kh in range(KH):
        for kw in range(KW):
            dX_p[:, :, kh:kh + H_out, kw:kw + W_out] += cols6[:, :, kh, kw, :, :]

    if pad > 0:
        return dX_p[:, :, pad:pad + H, pad:pad + W]
    return dX_p


def col2im(cols, input_shape, kernel_size, stride=1, pad=0):
    """
    Convert columns back to image (inverse of im2col).
    Dispatches to stride-1 fast path or general np.add.at version.

    Args:
        cols (ndarray): shape (C * KH * KW, B * H_out * W_out)
        input_shape (tuple): Original shape (B, C, H, W)
        kernel_size (int or tuple): Convolution kernel size
        stride (int): Stride used
        pad (int): Padding used

    Returns:
        ndarray: Reconstructed tensor of shape (B, C, H, W)
    """
    if stride == 1:
        return _col2im_stride1(cols, input_shape, kernel_size, pad)
    return _col2im_addat(cols, input_shape, kernel_size, stride, pad)


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError


def _param_pairs(layer):
    """Yield (weight_name, bias_name) pairs for a layer's trainable parameters."""
    if hasattr(layer, 'W') and hasattr(layer, 'b'):
        yield ('W', 'b')
    if hasattr(layer, 'W1') and hasattr(layer, 'b1'):
        yield ('W1', 'b1')
    if hasattr(layer, 'W2') and hasattr(layer, 'b2'):
        yield ('W2', 'b2')


def _param_layers(layers):
    """Yield layers that have trainable weights."""
    for layer in layers:
        if list(_param_pairs(layer)):
            yield layer


class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def step(self, layers):
        for layer in _param_layers(layers):
            for wname, bname in _param_pairs(layer):
                dw = getattr(layer, 'd' + wname)
                db = getattr(layer, 'd' + bname)
                setattr(layer, wname, getattr(layer, wname) - self.lr * dw)
                setattr(layer, bname, getattr(layer, bname) - self.lr * db)


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def _update_param(self, layer, pname):
        """Adam update for a single parameter (weight or bias)."""
        p = getattr(layer, pname)
        dp = getattr(layer, 'd' + pname)
        m_key = '_adam_m_' + pname
        v_key = '_adam_v_' + pname

        if not hasattr(layer, m_key):
            setattr(layer, m_key, np.zeros_like(p))
            setattr(layer, v_key, np.zeros_like(p))

        m = self.beta1 * getattr(layer, m_key) + (1 - self.beta1) * dp
        v = self.beta2 * getattr(layer, v_key) + (1 - self.beta2) * (dp ** 2)
        setattr(layer, m_key, m)
        setattr(layer, v_key, v)

        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        setattr(layer, pname, p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

    def step(self, layers):
        self.t += 1
        for layer in _param_layers(layers):
            for wname, bname in _param_pairs(layer):
                self._update_param(layer, wname)
                self._update_param(layer, bname)


class Network:
    def __init__(self, layers, optimizer=None):
        self.layers = layers
        self.optimizer = optimizer or SGD()

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dY):
        for layer in reversed(self.layers):
            dY = layer.backward(dY)
        return dY

    def step(self):
        self.optimizer.step(self.layers)


class ConvLayer(Layer):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, pad: int = 1):
        """
        Convolutional Layer using im2col for efficient computation.
        Uses CuPy for GPU matrix ops when available, falls back to NumPy on CPU.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of filters.
            kernel_size (int): Convolution kernel size (default 3).
            stride (int): Stride (default 1).
            pad (int): Zero-padding (default 1).
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

    def forward(self, X):
        B, C, H, W = X.shape
        F = self.out_channels

        H_out, W_out, patches_col = im2col(X, self.kernel_size, self.stride, self.pad)

        # Flatten filters to (F, C*KH*KW) and compute convolution as matrix multiply
        W_col = self.W.reshape(F, -1)
        out = W_col @ patches_col + self.b[:, None]

        out = out.reshape(F, B, H_out, W_out).transpose(1, 0, 2, 3)

        # Cache for backward pass
        self.patches = patches_col
        self.H_in = H
        self.W_in = W

        return out

    def backward(self, dY):
        B, F, H_out, W_out = dY.shape
        C = self.in_channels
        H, W = self.H_in, self.W_in

        # Flatten dY to (F, B*H_out*W_out) to match im2col layout
        dY_flat = dY.transpose(1, 0, 2, 3).reshape(F, -1)

        self.dW = (dY_flat @ self.patches.T).reshape(self.W.shape)
        self.db = dY.sum(axis=(0, 2, 3))

        # Propagate gradient back through input
        W_col = self.W.reshape(F, -1)
        dX_cols = W_col.T @ dY_flat

        return col2im(dX_cols, (B, C, H, W), self.kernel_size, self.stride, self.pad)


class ReLULayer(Layer):
    def forward(self, X):
        self.mask = (X > 0)
        return X * self.mask

    def backward(self, dY):
        return dY * self.mask


class MaxPoolLayer(Layer):
    def __init__(self, size=2, stride=2):
        """
        Args:
            size (int): Pooling window size (e.g., 2 for 2x2).
            stride (int): Stride step (e.g., 2 for non-overlapping).
        """
        super().__init__()
        self.size = size
        self.stride = stride

    def forward(self, X):
        self.X_shape = X.shape
        B, C, H, W = X.shape
        K = self.size

        H_out, W_out, cols = im2col(X, kernel_size=K, stride=self.stride, pad=0)
        self.H_out = H_out
        self.W_out = W_out

        # Reshape so each channel pools independently: (C, K*K, N)
        N = B * H_out * W_out
        cols_c = cols.reshape(C, K * K, N)

        self.max_idx = np.argmax(cols_c, axis=1)  # (C, N)
        out_flat = np.max(cols_c, axis=1)          # (C, N)
        self.N = N

        out = out_flat.reshape(C, B, H_out, W_out).transpose(1, 0, 2, 3)
        return out

    def backward(self, dY):
        B, C, H, W = self.X_shape
        K = self.size
        N = self.N

        dY_flat = dY.transpose(1, 0, 2, 3).reshape(C, N)

        # Scatter gradients to argmax positions
        dcols_c = np.zeros((C, K * K, N), dtype=dY.dtype)
        c_idx = np.arange(C)[:, None]
        n_idx = np.arange(N)[None, :]
        dcols_c[c_idx, self.max_idx, n_idx] = dY_flat

        dcols = dcols_c.reshape(C * K * K, N)
        return col2im(dcols, (B, C, H, W), kernel_size=K, stride=self.stride, pad=0)


class ReshapeLayer(Layer):
    def __init__(self, new_shape):
        """
        Args:
            new_shape (tuple): Target shape (excluding batch dimension).
        """
        self.new_shape = new_shape
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        B = X.shape[0]
        return X.reshape(B, *self.new_shape)

    def backward(self, dY):
        return dY.reshape(self.input_shape)


class GlobalContextLayer(Layer):
    """
    Squeeze-and-excite style global context: GAP → FC → FC → sigmoid → scale.
    Gives each spatial position knowledge of the whole image.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)  — same shape, channels re-weighted by global context.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        # FC1: C → mid
        scale = np.sqrt(2.0 / channels)
        self.W1 = np.random.randn(mid, channels) * scale
        self.b1 = np.zeros(mid)
        # FC2: mid → C
        scale2 = np.sqrt(2.0 / mid)
        self.W2 = np.random.randn(channels, mid) * scale2
        self.b2 = np.zeros(channels)

        self.channels = channels
        self.mid = mid

    def forward(self, X):
        B, C, H, W = X.shape
        self.X = X

        # Global Average Pooling: (B, C)
        self.gap = X.mean(axis=(2, 3))  # (B, C)

        # FC1 + ReLU: (B, C) → (B, mid)
        self.fc1_out = self.gap @ self.W1.T + self.b1  # (B, mid)
        self.relu_mask = (self.fc1_out > 0)
        self.fc1_relu = self.fc1_out * self.relu_mask  # (B, mid)

        # FC2 + Sigmoid: (B, mid) → (B, C)
        self.fc2_out = self.fc1_relu @ self.W2.T + self.b2  # (B, C)
        self.scale = 1.0 / (1.0 + np.exp(-self.fc2_out))  # sigmoid, (B, C)

        # Scale: (B, C, 1, 1) * (B, C, H, W)
        return X * self.scale[:, :, None, None]

    def backward(self, dY):
        B, C, H, W = dY.shape

        # dY wrt scaled output: out = X * scale
        # d(out)/d(X) = scale, d(out)/d(scale) = X
        dX_direct = dY * self.scale[:, :, None, None]

        # d(out)/d(scale): sum over spatial dims because scale is (B, C)
        dscale = (dY * self.X).sum(axis=(2, 3))  # (B, C)

        # Sigmoid backward: d(sigmoid)/d(fc2_out) = sig * (1 - sig)
        dsig = dscale * self.scale * (1.0 - self.scale)  # (B, C)

        # FC2 backward
        self.dW2 = dsig.T @ self.fc1_relu  # (C, mid)
        self.db2 = dsig.sum(axis=0)  # (C,)
        dfc1_relu = dsig @ self.W2  # (B, mid)

        # ReLU backward
        dfc1 = dfc1_relu * self.relu_mask  # (B, mid)

        # FC1 backward
        self.dW1 = dfc1.T @ self.gap  # (mid, C)
        self.db1 = dfc1.sum(axis=0)  # (mid,)
        dgap = dfc1 @ self.W1  # (B, C)

        # GAP backward: spread gradient evenly across spatial positions
        dX_gap = dgap[:, :, None, None] / (H * W)

        return dX_direct + dX_gap


class SoftmaxLayer(Layer):
    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis
        self.Y = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        X_shift = X - np.max(X, axis=self.axis, keepdims=True)
        eX = np.exp(X_shift)
        self.Y = eX / np.sum(eX, axis=self.axis, keepdims=True)
        return self.Y

    def backward(self, dY: np.ndarray) -> np.ndarray:
        dot = np.sum(dY * self.Y, axis=self.axis, keepdims=True)
        dX = self.Y * (dY - dot)
        return dX


class CrossEntropyLoss(Layer):
    def forward(self, X: np.ndarray, Y: np.ndarray, images: np.ndarray = None,
                eps_black: float = 0.05) -> float:
        """
        Args:
            X: Network output probabilities, shape (B, C, H, W).
            Y: Ground-truth permutation labels, shape (B, N^2).
            images: Original input images (B, 1, H_img, W_img). When provided,
                    black patches (all pixels < eps_black) are masked out of the loss.
            eps_black: Threshold below which a patch is considered black.
        """
        B, C, H, W = X.shape
        Y_hw = Y.reshape(B, H, W)

        # Move class dim last for gather indexing: (B, H, W, C)
        X_bhwc = X.transpose(0, 2, 3, 1)
        self.X_bhwc = X_bhwc
        self.Y_hw = Y_hw

        # Build per-patch mask: 1 for non-black, 0 for black (B, H, W, 1)
        if images is not None:
            _, _, H_img, W_img = images.shape
            ph, pw = H_img // H, W_img // W
            # Tile image into (B, H, ph, W, pw) and check if all pixels < eps
            im_tiles = images[:, 0].reshape(B, H, ph, W, pw)
            is_black = np.all(im_tiles < eps_black, axis=(2, 4))  # (B, H, W)
            mask = (~is_black).astype(np.float32)[..., None]
        else:
            mask = np.ones((B, H, W, 1), dtype=np.float32)

        self.mask = mask
        num_valid = float(max(mask.sum(), 1.0))
        self._num_valid = num_valid

        # Gather probability of the target class at each spatial position
        b_idx = np.arange(B)[:, None, None]
        h_idx = np.arange(H)[None, :, None]
        w_idx = np.arange(W)[None, None, :]
        target_p = X_bhwc[b_idx, h_idx, w_idx, Y_hw]

        loss = -np.sum(mask[..., 0] * np.log(target_p + 1e-12)) / num_valid
        return loss

    def backward(self) -> np.ndarray:
        X_bhwc = self.X_bhwc
        Y_hw = self.Y_hw
        mask = self.mask
        B, H, W, C = X_bhwc.shape

        dX_bhwc = np.zeros_like(X_bhwc, dtype=np.float32)

        b_idx = np.arange(B)[:, None, None]
        h_idx = np.arange(H)[None, :, None]
        w_idx = np.arange(W)[None, None, :]

        # Gradient of -log(p) at target class only
        target_p = X_bhwc[b_idx, h_idx, w_idx, Y_hw] + 1e-12
        dX_bhwc[b_idx, h_idx, w_idx, Y_hw] = -1.0 / target_p

        dX_bhwc *= (mask / self._num_valid)

        return dX_bhwc.transpose(0, 3, 1, 2)
