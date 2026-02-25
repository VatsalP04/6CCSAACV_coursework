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


def _param_layers(layers):
    """Yield layers that have trainable weights."""
    for layer in layers:
        if hasattr(layer, 'W') and hasattr(layer, 'b'):
            yield layer


class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def step(self, layers):
        for layer in _param_layers(layers):
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in _param_layers(layers):
            if not hasattr(layer, 'mW'):
                layer.mW = np.zeros_like(layer.W)
                layer.vW = np.zeros_like(layer.W)
                layer.mb = np.zeros_like(layer.b)
                layer.vb = np.zeros_like(layer.b)

            layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
            layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (layer.dW ** 2)
            mW_hat = layer.mW / (1 - self.beta1 ** self.t)
            vW_hat = layer.vW / (1 - self.beta2 ** self.t)
            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)

            layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db
            layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (layer.db ** 2)
            mb_hat = layer.mb / (1 - self.beta1 ** self.t)
            vb_hat = layer.vb / (1 - self.beta2 ** self.t)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


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
        """Forward pass: Y = W_col @ X_col + b"""
        B, C, H, W = X.shape
        F = self.out_channels

        H_out, W_out, patches_col = im2col(X, self.kernel_size, self.stride, self.pad)

        W_col = self.W.reshape(F, -1)

        # GPU matmul (or CPU fallback via xp=numpy)
        out = np.asarray(
            xp.matmul(xp.asarray(W_col), xp.asarray(patches_col))
            + xp.asarray(self.b[:, None])
        )

        out = out.reshape(F, B, H_out, W_out).transpose(1, 0, 2, 3)

        self.patches = patches_col
        self.H_in = H
        self.W_in = W

        return out

    def backward(self, dY):
        """Backward pass: compute dW, db, dX."""
        B, F, H_out, W_out = dY.shape
        C = self.in_channels
        H, W = self.H_in, self.W_in

        dY_reshaped = dY.transpose(1, 0, 2, 3).reshape(F, -1)

        # Gradient wrt weights
        self.dW = np.asarray(
            xp.matmul(xp.asarray(dY_reshaped), xp.asarray(self.patches.T))
        ).reshape(self.W.shape)

        # Gradient wrt bias
        self.db = np.asarray(xp.sum(xp.asarray(dY), axis=(0, 2, 3)))

        # Gradient wrt input
        W_col = self.W.reshape(F, -1)
        dX_patches = np.asarray(
            xp.matmul(xp.asarray(W_col.T), xp.asarray(dY_reshaped))
        )

        dX = col2im(dX_patches, (B, C, H, W), self.kernel_size, self.stride, self.pad)

        return dX


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
        self.X = X
        B, C, H, W = X.shape

        H_out, W_out, patches_col = im2col(X, self.size, self.stride)

        patches_reshaped = patches_col.reshape(C, self.size * self.size, -1)

        self.max_idx = np.argmax(patches_reshaped, axis=1)  # (C, N)
        self.rows = np.arange(C)[:, None]
        self.cols = np.arange(B * H_out * W_out)

        out_flat = patches_reshaped[self.rows, self.max_idx, self.cols]  # (C, N)
        out = out_flat.reshape(C, B, H_out, W_out).transpose(1, 0, 2, 3)

        self.H_in = H
        self.W_in = W

        return out

    def backward(self, dY):
        B, C, H_out, W_out = dY.shape
        H, W = self.H_in, self.W_in

        dpatches = np.zeros(
            (C, self.size * self.size, B * H_out * W_out),
            dtype=dY.dtype
        )

        dpatches[self.rows, self.max_idx, self.cols] = \
            dY.transpose(1, 0, 2, 3).reshape(C, B * H_out * W_out)

        dpatches_col = dpatches.reshape(C * self.size * self.size, -1)

        dX = col2im(dpatches_col, (B, C, H, W), kernel_size=self.size, stride=self.stride)

        return dX


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
                    black patches (mean < eps_black) are masked out of the loss.
            eps_black: Threshold below which a patch is considered black.
        """
        B, C, H, W = X.shape

        self.X_reshaped = np.transpose(X, (0, 2, 3, 1))  # (B, H, W, C)

        self.target_one_hot = np.zeros_like(self.X_reshaped, dtype=np.float32)
        self.target_one_hot[
            np.arange(B)[:, None, None],
            np.arange(H)[None, :, None],
            np.arange(W)[None, None, :],
            Y.reshape(B, H, W)
        ] = 1.0

        # Build per-patch mask: 1 for non-black, 0 for black
        if images is not None:
            _, _, H_img, W_img = images.shape
            ph, pw = H_img // H, W_img // W
            # mask shape (B, H, W)
            mask = np.ones((B, H, W), dtype=np.float32)
            for i in range(H):
                for j in range(W):
                    patch = images[:, 0, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                    # Match compute_reconstruction_accuracy: black if ALL pixels < eps
                    is_black = np.all(patch < eps_black, axis=(1, 2))  # (B,)
                    mask[:, i, j] = (~is_black).astype(np.float32)
            self.mask = mask[:, :, :, np.newaxis]  # (B, H, W, 1)
        else:
            self.mask = np.ones((B, H, W, 1), dtype=np.float32)

        num_valid = max(self.mask.sum(), 1.0)

        loss = -np.sum(
            self.mask * self.target_one_hot * np.log(self.X_reshaped + 1e-12)
        ) / num_valid

        self._num_valid = num_valid
        return loss

    def backward(self) -> np.ndarray:
        dX_reshaped = -self.mask * self.target_one_hot / (self.X_reshaped + 1e-12)
        dX_reshaped /= self._num_valid

        dX = np.transpose(dX_reshaped, (0, 3, 1, 2))
        return dX
