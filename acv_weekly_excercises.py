#!/usr/bin/env python
# coding: utf-8

# # Week 1

# In[20]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install cupy-cuda12x')

import os
from google.colab import drive


# In[21]:


import os
from google.colab import drive

def mount_google_drive(drive_path = "/content/drive"):
  """Check if Google Drive is mounted in Colab. Mount it if not."""
  if os.path.exists(drive_path):
    print("Google Drive is already mounted.")
  else:
    print("Mounting Google Drive...")
    drive.mount(drive_path)
    if os.path.exists(drive_path):
      print("Google Drive mounted successfully.")
    else:
      print("Failed to mount Google Drive. Please try again.")


# In[22]:


mount_google_drive()


# In[23]:


# import zipfile
# import os

# zip_path = '/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32.zip'
# extract_to = '/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets'

# # Make sure the extraction folder exists
# os.makedirs(extract_to, exist_ok=True)

# # Extract the zip
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#   zip_ref.extractall(extract_to)


# In[24]:


# mount_google_drive()

# def list_all_images(file_path):
#     """List all image filenames in a directory."""
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Directory not found: {file_path}")

#     images = []
#     for filename in os.listdir(file_path):
#         if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             images.append(filename)

#     if len(images) == 0:
#         print("No images found in the directory.")

#     return images


# In[25]:


# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# import random

# def display_N_images(N, file_path):
#   # Use random.sample instead of random.choice to select N unique images
#   random_imgs = random.sample(list_all_images(file_path),k=N)

#   # Create subplot
#   fig, axes = plt.subplots(1, N, figsize=(4*N, 4))

#   # If N == 1, axes is not a list
#   if N == 1:
#       axes = [axes]

#   for ax, filename in zip(axes, random_imgs):
#       path = os.path.join(file_path, filename)

#       # Load image using OpenCV
#       img = cv.imread(path)
#       assert img is not None, f"Could not read {filename}"

#       # Convert BGR (OpenCV) → RGB (Matplotlib)
#       img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#       # Display image
#       ax.imshow(img)
#       ax.set_title(filename)
#       ax.axis("off")   # hide ticks and labels

#   plt.tight_layout()
#   plt.show()

# display_N_images(10, "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32")


# In[26]:


# import random
# import json
# import os # Import os module for directory operations

# def cross_val_imgs(SEED, K, file_path, output_jsons):
#     os.makedirs(output_jsons, exist_ok=True)

#     all_images = list_all_images(file_path)
#     if len(all_images) < K:
#         raise ValueError(f"Need at least {K} images, found {len(all_images)}")

#     random.seed(SEED)
#     random.shuffle(all_images)

#     n = len(all_images)
#     fold_size = n // K

#     for i in range(K):
#         val_start = i * fold_size
#         val_end = (i + 1) * fold_size if i < K - 1 else n

#         val_files = all_images[val_start:val_end]
#         train_files = all_images[:val_start] + all_images[val_end:]

#         fold_data = {"train": train_files, "val": val_files}

#         fold_path = os.path.join(output_jsons, f"fold_{i+1}.json")
#         with open(fold_path, "w") as f:
#             json.dump(fold_data, f, indent=4)

#         print("Saved:", fold_path)

# cross_val_imgs(SEED = 2025, K = 5, file_path = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32", output_jsons= "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32_cross_val")
# print(os.listdir("/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32_cross_val"))


# In[27]:


import random
import json
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
from google.colab import drive
from typing import Tuple, List # Import Tuple and List for type hinting

class PatchShuffleDataLoader:
  def __init__(self, json_file: str, dataset_dir: str, batch_size: int = 8, image_size: Tuple[int, int] = (32, 32), num_patches: int = 4,shuffle: bool = True):
      """
      DataLoader with patch shuffling.
      Labels are permutation vectors of length num_patches^2.
      Args:
        json_file (str): Path to JSON file (contains "train" and "val" lists).
        dataset_dir (str): Root dataset directory containing the images.
        batch_size (int): Number of samples per batch (default=8).
        image_size (Tuple[int,int]): Resize images to this size (W, H).
        Default is (32, 32).
        num_patches (int): Number of patches along one dimension (default=4).
        shuffle (bool): Shuffle dataset each epoch if True (default=True).
      """
      self.dataset_dir = dataset_dir
      self.batch_size = batch_size
      self.image_size = image_size
      self.num_patches = num_patches
      self.shuffle = shuffle

      # Load train and validation files from the JSON file
      with open(json_file, 'r') as f:
          data = json.load(f)
      self.train_files = data['train']
      self.val_files = data['val']

  def _shuffle_patches(self, image: np.ndarray, indices: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
      """
      Divide image into patches, shuffle, reconstruct and return permuted image and permutation labels.
      Parameters:
          image (np.ndarray): Input image of shape (H, W, C)
          indices (np.ndarray, optional): Predefined permutation of patch indices.
                                          If None, a random shuffle is used.
      Returns:
          Tuple[np.ndarray, np.ndarray]: Shuffled image and the permutation indices
      """
      # (a) Calculate dimensions
      H, W = image.shape[0], image.shape[1]
      N = self.num_patches  # patches per dimension
      ph, pw = H // N, W // N

      # Safety checks: ensure exact tiling
      if H % N != 0 or W % N != 0:
          raise ValueError(
              f"Image size ({H}, {W}) not divisible by num_patches={N}. "
              "Choose a different num_patches or resize/crop the image."
          )

      # Store patch size if you want it as a class attribute
      self.patch_size = (ph, pw)

      # (b) Extract patches (flatten into 1D list in row-major order)
      patches = []
      for i in range(N):          # rows
          for j in range(N):      # cols
              patch = image[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
              patches.append(patch)
      patches = np.array(patches)  # shape: (N^2, ph, pw, C)

      # (c) Handle indices (permutation)
      num_total = N * N
      if indices is None:
          indices = np.random.permutation(num_total)
      else:
          indices = np.asarray(indices)
          if indices.shape != (num_total,):
              raise ValueError(f"indices must have shape ({num_total},), got {indices.shape}")
          if set(indices.tolist()) != set(range(num_total)):
              raise ValueError("indices must be a permutation of 0..N^2-1 (no repeats, none missing)")

      # (d) Reconstruct shuffled image + build label mapping
      shuffled_image = np.empty_like(image)
      label = np.empty(num_total, dtype=np.int64)

      for new_pos, original_id in enumerate(indices):
          # new_pos -> (i, j) in the output grid
          out_i = new_pos // N
          out_j = new_pos % N

          # place the chosen original patch into this output location
          shuffled_image[out_i*ph:(out_i+1)*ph, out_j*pw:(out_j+1)*pw, :] = patches[original_id]

          # label maps original patch id -> where it ended up
          label[original_id] = new_pos

      # (e) Return
      return shuffled_image, label



  def _load_image(self, filepath: str)-> Tuple[np.ndarray, np.ndarray]:
      """
      Load and preprocess a single image with patch shuffling
      (CHW + permutation labels).

      -> Tuple[np.ndarray, np.ndarray]:
      """
       # (a) File I/O: construct full path and load image
      full_path = os.path.join(self.dataset_dir, filepath)
      image = cv.imread(full_path)

      if image is None:
          raise ValueError(f"Failed to load image: {full_path}")

      # (b) Color conversion: BGR -> grayscale
      image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

      # (c) Resize to target image size
      image = cv.resize(image, self.image_size)

      # (d) Dimension check: ensure (H, W, 1)
      if image.ndim == 2:
          image = image[:, :, np.newaxis]

      # (e) Patch shuffle: get shuffled image and permutation labels
      image, labels = self._shuffle_patches(image)

      # (f) Normalisation: float32 and scale to [0, 1]
      image = image.astype(np.float32) / 255.0

      # (g) Transpose: Channel-Last (H, W, C) -> Channel-First (C, H, W)
      image = np.transpose(image, (2, 0, 1))

      # (h) Return processed image and labels
      return image, labels



  def _get_batches(self, file_list: List[str]):
      """ Yield mini-batches of data and patch permutation labels.
      Args:
          file_list (List[str]): List of image filenames to process.
      Yields:
          Tuple[np.ndarray, np.ndarray]:
            - Batch of images: Shape (B, C, H, W)
            - Batch of labels: Shape (B, N^2) where N is num_patches
      """
      # (a) Shuffle
      files = list(file_list)  # avoid modifying the original list
      if getattr(self, "shuffle", False):
          random.shuffle(files)

      # (b) Batch loop
      for start in range(0, len(files), self.batch_size):
          batch_files = files[start:start + self.batch_size]

          X, Y = [], []

          # Image processing loop
          for fname in batch_files:
              img, label = self._load_image(fname)  # img: (C,H,W), label: (N^2,)
              X.append(img)
              Y.append(label)

          # Convert to NumPy
          X = np.stack(X, axis=0)  # (B,C,H,W)
          Y = np.stack(Y, axis=0)  # (B,N^2)

          # Yield batch
          yield X, Y

  def train_batches(self):
    return self._get_batches(self.train_files)

  def val_batches(self):
    return self._get_batches(self.val_files)


  def display_training_image(self, image: np.ndarray):
    # CHW -> HWC
    new_im = np.transpose(image, (1, 2, 0))
    # De-normalize back to [0, 255]
    new_im = np.clip(new_im * 255.0, 0, 255).astype(np.uint8)
    # Show image
    plt.imshow(new_im)
    plt.axis("off")
    plt.show()

  def display_reconstructed_training_image(self, image: np.ndarray,label: np.ndarray):
    # CHW -> HWC
    new_im = np.transpose(image, (1, 2, 0))
    # De-normalize back to [0, 255]
    new_im = np.clip(new_im * 255.0, 0, 255).astype(np.uint8)
    # Bring the patches back to their original position
    new_im, _ = self._shuffle_patches(new_im, label)
    # Show image
    plt.imshow(new_im)
    plt.axis("off")
    plt.show()


# if __name__ == "__main__":
#   # Mount Google Drive (if not mounted yet)
#   mount_google_drive()

#   # Create dataloader
#   dataset_dir = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32"
#   json_file = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32_cross_val/fold_1.json"
#   loader = PatchShuffleDataLoader(json_file, dataset_dir, batch_size=8)

#   # Loop over training batches (just one iteration for demo)
#   for X_batch, Y_batch in loader.train_batches():
#     print("Batch images shape:", X_batch.shape) # (8, 1, 32, 32)
#     print("Batch labels shape:", Y_batch.shape) # (8, 16)
#     for i in range(loader.batch_size):
#       print(f"Training image {i}:")
#       loader.display_training_image(X_batch[i])
#       print(f"Original image {i}:")
#       loader.display_reconstructed_training_image(X_batch[i], Y_batch[i])
#     break


# # Week 2
# 

# In[28]:


import numpy as np

def im2col(X, kernel_size, stride=1, pad=0):
    """
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

    # kernel_size handling
    if isinstance(kernel_size, int):
        KH, KW = kernel_size, kernel_size
    else:
        KH, KW = kernel_size  # tuple like (KH, KW)

    # 1) pad (pad only H and W dims)
    if pad > 0:
        X_p = np.pad(
            X,
            pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0
        )
    else:
        X_p = X

    H_p, W_p = X_p.shape[2], X_p.shape[3]

    # 2) output spatial size
    H_out = (H_p - KH) // stride + 1
    W_out = (W_p - KW) // stride + 1

    # 3) allocate cols: each column is one flattened patch
    patch_size = C * KH * KW
    n_patches_per_img = H_out * W_out
    cols = np.zeros((patch_size, B * n_patches_per_img), dtype=X.dtype)

    # 4) extract patches and fill cols
    col_idx = 0
    for b in range(B):
        for i in range(H_out):
            h0 = i * stride
            h1 = h0 + KH
            for j in range(W_out):
                w0 = j * stride
                w1 = w0 + KW

                patch = X_p[b, :, h0:h1, w0:w1]     # (C, KH, KW)
                cols[:, col_idx] = patch.reshape(-1) # (C*KH*KW,)
                col_idx += 1

    return H_out, W_out, cols

import numpy as np

def col2im(cols, input_shape, kernel_size, stride=1, pad=0):
    """
    Args:
      cols (ndarray): 2D array of shape (C * KH * KW, B * H_out * W_out).
      input_shape (tuple): Original shape (B, C, H, W).
      kernel_size (int or tuple): Size of the convolution kernel.
      stride (int): Stride used.
      pad (int): Padding used.

    Returns:
      ndarray: The reconstructed tensor of shape (B, C, H, W).
    """
    B, C, H, W = input_shape

    # kernel_size handling
    if isinstance(kernel_size, int):
        KH, KW = kernel_size, kernel_size
    else:
        KH, KW = kernel_size

    # padded spatial dims
    H_p = H + 2 * pad
    W_p = W + 2 * pad

    # infer output spatial dims (must match what im2col produced)
    H_out = (H_p - KH) // stride + 1
    W_out = (W_p - KW) // stride + 1

    # we'll reconstruct into padded grad tensor
    dX_p = np.zeros((B, C, H_p, W_p), dtype=cols.dtype)

    col_idx = 0
    for b in range(B):
        for i in range(H_out):
            h0 = i * stride
            h1 = h0 + KH
            for j in range(W_out):
                w0 = j * stride
                w1 = w0 + KW

                patch = cols[:, col_idx].reshape(C, KH, KW)
                dX_p[b, :, h0:h1, w0:w1] += patch   # IMPORTANT: += accumulates overlaps
                col_idx += 1

    # remove padding
    if pad > 0:
        return dX_p[:, :, pad:pad+H, pad:pad+W]
    else:
        return dX_p




# In[29]:


import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError

class Network:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def backward(self, dY):
    for layer in reversed(self.layers):
      dY = layer.backward(dY)
    return dY

  def step(self, learning_rate):
    for layer in self.layers:
      if hasattr(layer, 'W') and hasattr (layer,'b'):
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db



# In[30]:


import numpy as np
import cupy as cp

class ConvLayer(Layer):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, pad: int = 1):
        """
        Convolutional Layer

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            out_channels (int): Number of filters.
            kernel_size (int): Size of the convolution kernel (default 3).
            stride (int): Stride size (default 1).
            pad (int): Zero-padding size (default 1).
        """

        super().__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        self.W = np.random.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ) * scale

        # Bias per filter
        self.b = np.zeros(out_channels)


    def forward(self, X):
        """
        Forward pass using im2col:

        Y_col = W_col @ X_col + b
        """

        B, C, H, W = X.shape
        F, _, KH, KW = self.W.shape

        # Convert image to column matrix
        H_out, W_out, patches_col = im2col(
            X,
            self.kernel_size,
            self.stride,
            self.pad
        )

        # Reshape filters
        W_col = self.W.reshape(F, -1)

        # Matrix multiplication (GPU)
        out = cp.asnumpy(
            cp.matmul(cp.asarray(W_col),
                      cp.asarray(patches_col))
            + cp.asarray(self.b[:, None])
        )

        # Reshape back to (B, F, H_out, W_out)
        out = out.reshape(F, B, H_out, W_out).transpose(1, 0, 2, 3)

        # Store for backward pass
        self.patches = patches_col
        self.H_in = H
        self.W_in = W

        return out


    def backward(self, dY):
        """
        Backward pass.

        Computes:
        - dW (gradient wrt filters)
        - db (gradient wrt bias)
        - dX (gradient wrt input)
        """

        B, F, H_out, W_out = dY.shape
        C = self.in_channels
        KH = KW = self.kernel_size
        H, W = self.H_in, self.W_in

        # Reshape upstream gradient
        dY_reshaped = dY.transpose(1, 0, 2, 3).reshape(F, -1)

        # ---- Gradient wrt weights ----
        self.dW = cp.asnumpy(
            cp.matmul(cp.asarray(dY_reshaped),
                      cp.asarray(self.patches.T))
        )

        self.dW = self.dW.reshape(self.W.shape)

        # ---- Gradient wrt bias ----
        self.db = cp.asnumpy(
            cp.sum(cp.asarray(dY), axis=(0, 2, 3))
        )

        # ---- Gradient wrt input ----
        W_col = self.W.reshape(F, -1)

        dX_patches = cp.asnumpy(
            cp.matmul(cp.asarray(W_col.T),
                      cp.asarray(dY_reshaped))
        )

        dX = col2im(
            dX_patches,
            (B, C, H, W),
            self.kernel_size,
            self.stride,
            self.pad
        )

        return dX


# In[31]:


import numpy as np
import torch


def test_conv_layer():

    # Parameters
    B, C, H, W = 2, 3, 5, 5          # Batch size, in_channels, height, width
    F = 4                            # out_channels
    kernel_size = 3
    stride = 1
    pad = 1

    # Dummy input
    X_np = np.random.randn(B, C, H, W).astype(np.float32)
    X_torch = torch.tensor(X_np, requires_grad=True)

    # Custom ConvLayer
    conv_np = ConvLayer(
        C,
        F,
        kernel_size=kernel_size,
        stride=stride,
        pad=pad
    )

    # PyTorch Conv2d
    conv_torch = torch.nn.Conv2d(
        in_channels=C,
        out_channels=F,
        kernel_size=kernel_size,
        stride=stride,
        padding=pad,
        bias=True
    )

    # Copy weights & biases from custom layer -> PyTorch
    with torch.no_grad():
        conv_torch.weight.copy_(torch.tensor(conv_np.W))
        conv_torch.bias.copy_(torch.tensor(conv_np.b))

    # ----------------------
    # Forward pass
    # ----------------------
    out_np = conv_np.forward(X_np)
    out_torch = conv_torch(X_torch)

    assert np.allclose(
        out_np,
        out_torch.detach().numpy(),
        atol=1e-5
    ), "[ERROR] Convolutional layer: Forward outputs do not match PyTorch."

    # ----------------------
    # Backward pass
    # ----------------------
    dY_np = np.random.randn(*out_np.shape).astype(np.float32)
    dY_torch = torch.tensor(dY_np)

    # Custom backward
    dX_np = conv_np.backward(dY_np)

    # PyTorch backward
    out_torch.backward(dY_torch)
    dX_torch = X_torch.grad.detach().numpy()

    # ----------------------
    # Gradient comparisons
    # ----------------------
    assert np.allclose(
        conv_np.dW,
        conv_torch.weight.grad.detach().numpy(),
        atol=1e-5
    ), "[ERROR] Convolutional layer: dW mismatch."

    assert np.allclose(
        conv_np.db,
        conv_torch.bias.grad.detach().numpy(),
        atol=1e-5
    ), "[ERROR] Convolutional layer: db mismatch."

    assert np.allclose(
        dX_np,
        dX_torch,
        atol=1e-5
    ), "[ERROR] Convolutional layer: dX mismatch."

    print("[INFO] Convolution layer tests passed!")


if __name__ == "__main__":

    # Mount Google Drive
    mount_google_drive()

    # ----------------------
    # Configuration parameters
    # ----------------------

    dataset_dir = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32"
    json_file = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32_cross_val/fold_1.json"

    batch_size = 32
    image_size = (32, 32)
    num_patches = 4
    shuffle = True

    loader = PatchShuffleDataLoader(
        json_file,
        dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_patches=num_patches,
        shuffle=shuffle
    )

    # Run unit tests
    # test_conv_layer()


# # Week 3
# 

# In[32]:


class ReLULayer(Layer):
    def forward(self, X):
        self.mask = (X > 0)
        return X * self.mask  # zeros out negatives

    def backward(self, dY):
        return dY * self.mask


# In[ ]:


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
        """
        Use im2col -> reshape -> max/argmax -> output.
        """
        self.X = X  # Store input

        B, C, H, W = X.shape
        # Convert image to column matrix
        H_out, W_out, patches_col = im2col(
            X,
            self.size,
            self.stride
        )
        # Reshape to (C, size*size, B*H_out*W_out)
        patches_reshaped = patches_col.reshape(C, self.size * self.size, -1)

        # Store argmax for backward pass
        self.max_idx = np.argmax(patches_reshaped, axis=1) # shape: (C, N)

        # Build indexing helpers
        self.rows = np.arange(C)[:, None]       # channel indexes, shape: (C, 1)
        self.cols = np.arange(B*H_out*W_out)    # spatial + batch positions

        # Gather max values using advanced indexing
        out_flat = patches_reshaped[self.rows, self.max_idx, self.cols]
        # shape: (C, N)

        # Reshape back to image format
        out = out_flat.reshape(C, B, H_out, W_out).transpose(1, 0, 2, 3)
        # final shape: (B, C, H_out, W_out)        

        self.H_in = H
        self.W_in = W

        return out

    def backward(self, dY):
        B, C, H_out, W_out = dY.shape
        H, W = self.H_in, self.W_in

        # Initialize gradients to zero with the same shape as reshaped patches
        dpatches = np.zeros(
            (C, self.size * self.size, B * H_out * W_out),
            dtype=dY.dtype
        )

        # Place gradients in max positions
        dpatches[self.rows, self.max_idx, self.cols] = \
            dY.transpose(1, 0, 2, 3).reshape(C, B * H_out * W_out)

        # Flatten back to column form (C * KH * KW, B * H_out * W_out)
        dpatches_col = dpatches.reshape(C * self.size * self.size, -1)

        # Convert back to input shape with col2im
        dX = col2im(
            dpatches_col,
            (B, C, H, W),
            kernel_size=self.size,
            stride=self.stride
        )

        return dX


# In[34]:


class ReshapeLayer(Layer):
    def __init__(self, new_shape):
        """
        Args:
        new_shape (tuple): Target shape for the output (excluding batch dimension).
        """
        self.new_shape = new_shape
        self.input_shape = None  # To cache original shape for backward pass
        
    def forward(self, X):
        """
        Reshape X to (Batch, *new_shape). Cache original shape.
        """
        self.input_shape = X.shape
        B = X.shape[0]
        return X.reshape(B, *self.new_shape) 
        
    def backward(self, dY):
        """
        Reshape dY back to the original cached input_shape.
        """
        return dY.reshape(self.input_shape) 
        


# In[35]:


import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis
        self.Y = None  # cache for backward

    def forward(self, X: np.ndarray) -> np.ndarray:
        # stable softmax
        X_shift = X - np.max(X, axis=self.axis, keepdims=True)
        eX = np.exp(X_shift)
        self.Y = eX / np.sum(eX, axis=self.axis, keepdims=True)
        return self.Y

    def backward(self, dY: np.ndarray) -> np.ndarray:
        # Jacobian-vector product: dX = Y * (dY - sum(dY*Y))
        dot = np.sum(dY * self.Y, axis=self.axis, keepdims=True)
        dX = self.Y * (dY - dot)
        return dX


# In[36]:


import numpy as np

class CrossEntropyLoss(Layer):
    def forward(self, X: np.ndarray, Y: np.ndarray) -> float:
        B, C, H, W = X.shape

        # Move channel dimension last: (B, H, W, C)
        self.X_reshaped = np.transpose(X, (0, 2, 3, 1))

        # Create one-hot targets
        self.target_one_hot = np.zeros_like(self.X_reshaped, dtype=np.float32)
        self.target_one_hot[
            np.arange(B)[:, None, None],
            np.arange(H)[None, :, None],
            np.arange(W)[None, None, :],
            Y.reshape(B, H, W)
        ] = 1.0

        # Cross-entropy loss (averaged)
        loss = -np.sum(
            self.target_one_hot * np.log(self.X_reshaped + 1e-12)
        ) / (B * H * W)

        return loss

    def backward(self) -> np.ndarray:
        B, H, W, _ = self.X_reshaped.shape

        dX_reshaped = -self.target_one_hot / (self.X_reshaped + 1e-12)
        dX_reshaped /= (B * H * W)

        # Restore to (B, C, H, W)
        dX = np.transpose(dX_reshaped, (0, 3, 1, 2))
        return dX


# In[ ]:


def test_relu_layer():
    import torch

    # Parameters
    B, C, H, W = 2, 3, 8, 8

    # Dummy input
    X_np = np.random.randn(B, C, H, W).astype(np.float32)
    X_torch = torch.tensor(X_np, requires_grad=True)

    # Layers
    relu_np = ReLULayer()
    relu_torch = torch.nn.ReLU()

    # Forward pass
    out_np = relu_np.forward(X_np)
    out_torch = relu_torch(X_torch)

    # Compare forward
    assert np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5), \
        "[ ERROR ] ReLU layer: Forward outputs do not match PyTorch."

    # Backward pass
    dY_np = np.random.randn(*out_np.shape).astype(np.float32)
    dY_torch = torch.tensor(dY_np)

    dX_np = relu_np.backward(dY_np)
    out_torch.backward(dY_torch)
    dX_torch = X_torch.grad.detach().numpy()

    # Compare gradients
    assert np.allclose(dX_np, dX_torch, atol=1e-5), \
        "[ ERROR ] ReLU layer: dX mismatch."

    print("[ INFO ] ReLU layer tests passed!")

def test_maxpool_layer():
    import torch

    B, C, H, W = 2, 3, 8, 8
    kernel_size = 2
    stride = 2

    # Dummy input
    X_np = np.random.randn(B, C, H, W).astype(np.float32)
    X_torch = torch.tensor(X_np, requires_grad=True)

    # Layers
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
    import torch

    B, C, H, W = 2, 4, 3, 3

    # Create logits → convert to probabilities
    logits = np.random.randn(B * H * W, C).astype(np.float32)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    logprobs = np.log(probs)

    # Targets
    targets = np.random.randint(0, C, size=(B * H * W,))

    logprobs_torch = torch.tensor(logprobs, requires_grad=True)
    targets_torch = torch.tensor(targets, dtype=torch.long)

    # Reshape for custom layer
    probs_np = probs.reshape(B * H * W, 1, 1, C).transpose(0, 3, 1, 2)
    targets_np = targets.reshape(B * H * W, 1)

    # PyTorch NLLLoss
    ce_torch = torch.nn.NLLLoss()
    loss_torch = ce_torch(logprobs_torch, targets_torch)

    # Custom CrossEntropy
    ce_np = CrossEntropyLoss()
    loss_np = ce_np.forward(probs_np, targets_np)

    assert np.allclose(loss_np, loss_torch.item(), atol=1e-5), \
        "[ ERROR ] CrossEntropyLoss: Forward outputs do not match PyTorch."

    # Backward PyTorch
    loss_torch.backward()
    dX_torch_probs = logprobs_torch.grad.detach().numpy() / (probs + 1e-12)

    # Backward custom
    dX_np = ce_np.backward().transpose(0, 2, 3, 1).reshape(B * H * W, C)

    assert np.allclose(dX_np, dX_torch_probs, atol=1e-5), \
        "[ ERROR ] CrossEntropyLoss: Backward gradients mismatch."

    print("[ INFO ] CrossEntropyLoss layer tests passed!")

if __name__ == "__main__":
    mount_google_drive()

    test_conv_layer()
    test_relu_layer()
    test_maxpool_layer()
    test_crossentropy_loss_layer()

    # Configuration
    dataset_dir = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32"
    json_file = "/content/drive/MyDrive/Colab Notebooks/acv_exercises/datasets/acv_train_32x32_cross_val/fold_1.json"

    batch_size = 32
    image_size = (32, 32)
    num_patches = 4
    shuffle = True

    loader = PatchShuffleDataLoader(
        json_file,
        dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_patches=num_patches,
        shuffle=shuffle
    )


# # Week 4

# In[ ]:


# Create a convolutional neural network
net = Network([
    ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    MaxPoolLayer(size=2, stride=2),  # 16x16

    ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    MaxPoolLayer(size=2, stride=2),  # 8x8

    ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    MaxPoolLayer(size=2, stride=2),  # 4x4

    ConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    ConvLayer(in_channels=64, out_channels=16, kernel_size=1, stride=1, pad=0),

    SoftmaxLayer(axis=1)  # Apply softmax across channel dimension
])


# In[ ]:


import numpy as np
import scipy.optimize


def assign_patches(probs):
    """
    Solve the patch assignment problem using predicted probabilities.

    Args:
        probs: np.ndarray of shape (C, H, W),
               where C = H * W,
               probs[i, h, w] = probability that patch (h, w)
               was originally at position i.

    Returns:
        np.ndarray of shape (C,) containing assigned original
        position index for each patch.
    """

    C, H, W = probs.shape
    assert C == H * W, "[ ERROR ] Invalid shape for probabilities."

    # Build cost matrix (negative for maximization)
    cost = -probs.transpose(1, 2, 0).reshape(-1, C)

    # Hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

    assignment = np.zeros(C, dtype=int)
    assignment[row_ind] = col_ind

    return assignment


# In[ ]:


def compute_reconstruction_accuracy(im, pred, gt, num_patches: int = 4, eps: float = 0.05):
    """
    Compute proportion of correctly reconstructed patches,
    ignoring completely black patches.
    """

    C, H, W = im.shape
    assert C == 1, "[ ERROR ] Only grayscale images supported."

    assert pred.shape == gt.shape, "[ ERROR ] Prediction and ground truth must have same shape."
    assert len(pred.shape) == 1, "[ ERROR ] Predictions and ground truth must be flat vectors."

    # Patch dimensions
    ph = H // num_patches
    pw = W // num_patches
    assert ph == pw, "[ ERROR ] Only square patches supported."

    # Convert to 2D
    im_2d = np.squeeze(im)

    num_correct = 0
    num_total = 0
    i = -1

    for y in range(0, H, ph):
        for x in range(0, W, pw):
            i += 1

            # Ignore completely black patches
            if np.all(im_2d[y:y+ph, x:x+pw] < eps):
                continue

            num_total += 1

            if pred[i] == gt[i]:
                num_correct += 1

    return num_correct / num_total


# # Week 5

# In[ ]:


def compute_total_receptive_field(net):
    """
    Compute the receptive field of the whole network.
    Only considers ConvLayer and MaxPoolLayer.
    Returns the final receptive field size (assuming square receptive field).
    """

    R = 1  # Receptive field of input (1 pixel)
    j = 1  # Jump between receptive field centers

    for layer in net.layers:

        if isinstance(layer, ConvLayer):
            k = layer.kernel_size
            s = layer.stride
            p = layer.pad

            R += (k - 1) * j
            j = j * s

        elif isinstance(layer, MaxPoolLayer):
            k = layer.size
            s = layer.stride

            R += (k - 1) * j
            j = j * s

        else:
            # Ignore activations and Softmax
            continue

    return R


# In[ ]:


# Training parameters
num_epochs = 50
learning_rate = 1e-3


# Define convolutional network (if we could not load it from file)
net = Network([

    ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    MaxPoolLayer(size=2, stride=2),   # 16x16

    ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    MaxPoolLayer(size=2, stride=2),   # 8x8

    ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    MaxPoolLayer(size=2, stride=2),   # 4x4

    ConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, pad=1),
    ReLULayer(),

    ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad=1),

    ConvLayer(in_channels=64, out_channels=16, kernel_size=1, stride=1, pad=0),

    SoftmaxLayer(axis=1),  # across channels/classes
])


# Loop over epochs
for epoch in range(num_epochs):

    iter_train = 0

    # Loop over training batches
    for X_batch, Y_batch in loader.train_batches():
        # X_batch: (B, C, H, W)
        # Y_batch: (B, 16)

        iter_train += 1

        # Forward pass
        out_batch = net.forward(X_batch)   # (B, C, H, W)

        # Compute loss
        loss_layer = CrossEntropyLoss()
        loss = loss_layer.forward(out_batch, Y_batch)

        print(f"[INFO] Mode: Training, Epoch: {epoch}, "
              f"Iteration: {iter_train}, Loss: {loss}")

        # Backward pass
        dX = loss_layer.backward()
        net.backward(dX)

        # Update weights and biases with gradient descent
        net.step(learning_rate)

