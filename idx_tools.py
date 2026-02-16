import gzip
import struct

import numpy as np

# IDX file format magic numbers
IDX3_IMAGES_MAGIC = 2051
IDX1_LABELS_MAGIC = 2049


def open_file(path):
    """Open a file, handling gzip compression automatically."""
    path_str = str(path)
    return gzip.open(path, "rb") if path_str.endswith(".gz") else open(path, "rb")


def read_idx_images(path):
    """Read IDX3 format image data and return as numpy array (a 3D matrix, really)."""
    with open_file(path) as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        if magic != IDX3_IMAGES_MAGIC:
            raise ValueError(
                f"Invalid magic number {magic}, expected {IDX3_IMAGES_MAGIC} for IDX3 images"
            )
        
        data = f.read(num_images * rows * cols)
    
    images = np.frombuffer(data, dtype=np.uint8).reshape(num_images, rows, cols)
    print(f"Read {len(images)} images with shape {images.shape}.")
    return images


def read_idx_labels(path):
    """Read IDX1 format label data and return as numpy array."""
    with open_file(path) as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        
        if magic != IDX1_LABELS_MAGIC:
            raise ValueError(
                f"Invalid magic number {magic}, expected {IDX1_LABELS_MAGIC} for IDX1 labels"
            )
        
        data = f.read(num_labels)
    
    labels = np.frombuffer(data, dtype=np.uint8)
    print(f"Read {len(labels)} labels.")
    return labels


def read_emnist_data(labels_file, images_file):
    """Read label and image data and return as numpy arrays."""
    labels = read_idx_labels(labels_file)
    images = read_idx_images(images_file)

    return labels, images


def display_image(image):
    """Display a 28x28 grayscale image in the terminal using Unicode block characters."""
    # Unicode block characters from darkest to lightest
    chars = " ░▒▓█"
    
    # Image data must be transposed to see it "right"
    transformed = image.T
    
    for row in transformed:
        line = ""
        for pixel in row:
            # Map pixel value (0-255) to character index (0-4)
            # Convert to int to avoid uint8 overflow
            char_index = min(int(pixel) * len(chars) // 256, len(chars) - 1)
            line += chars[char_index] * 2  # Double width for better aspect ratio
        print(line)
    print()  # Empty line after image


def serialize_to_csv(image):
    """Serialize image matrix to comma-separated string."""
    return ",".join(str(int(pixel)) for pixel in image.flatten())
