import struct, gzip
import numpy as np

def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")

def read_idx_images(path):
    with _open(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic {magic}, expected 2051 (idx3 images)")
        data = f.read(n * rows * cols)
    return np.frombuffer(data, dtype=np.uint8).reshape(n, rows, cols)

def read_idx_labels(path):
    with _open(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic {magic}, expected 2049 (idx1 labels)")
        data = f.read(n)
    return np.frombuffer(data, dtype=np.uint8)

data = read_idx_images("dataset/gzip/emnist-mnist-test-images-idx3-ubyte.gz")
labels = read_idx_labels("dataset/gzip/emnist-mnist-test-labels-idx1-ubyte.gz")