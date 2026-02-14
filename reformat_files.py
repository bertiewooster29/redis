import gzip
import struct
from pathlib import Path

import numpy as np
import redis

# IDX file format magic numbers
IDX3_IMAGES_MAGIC = 2051
IDX1_LABELS_MAGIC = 2049


def open_file(path):
    """Open a file, handling gzip compression automatically."""
    path_str = str(path)
    return gzip.open(path, "rb") if path_str.endswith(".gz") else open(path, "rb")


def read_idx_images(path):
    """Read IDX3 format image data and return as numpy array."""
    with open_file(path) as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        if magic != IDX3_IMAGES_MAGIC:
            raise ValueError(
                f"Invalid magic number {magic}, expected {IDX3_IMAGES_MAGIC} for IDX3 images"
            )
        
        data = f.read(num_images * rows * cols)
    
    return np.frombuffer(data, dtype=np.uint8).reshape(num_images, rows, cols)


def read_idx_labels(path):
    """Read IDX1 format label data and return as numpy array."""
    with open_file(path) as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        
        if magic != IDX1_LABELS_MAGIC:
            raise ValueError(
                f"Invalid magic number {magic}, expected {IDX1_LABELS_MAGIC} for IDX1 labels"
            )
        
        data = f.read(num_labels)
    
    return np.frombuffer(data, dtype=np.uint8)


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


def serialize_to_binary(image):
    """Serialize image matrix to binary string (one byte per pixel)."""
    return image.tobytes()


def binary_to_csv(binary_data):
    """Convert binary serialized image to CSV format."""
    return ",".join(str(byte) for byte in binary_data)


def csv_to_binary(csv_string):
    """Convert CSV serialized image to binary format."""
    values = [int(val) for val in csv_string.split(",")]
    return bytes(values)


def store_samples_in_redis_as_json(redis_client, labels, images):
    """Store label and image data in Redis as JSON objects.
    
    Args:
        redis_client: Redis connection object
        labels: List/array of labels
        images: List/array of image matrices in original format
    """
    for i, (label, image) in enumerate(zip(labels, images)):
        # Convert image to CSV format
        csv_string = serialize_to_csv(image)
        
        # Convert CSV string to list of integers
        pixels = [int(val) for val in csv_string.split(",")]
        
        # Create JSON object
        data = {
            "label": int(label),
            "pixels": pixels
        }
        
        # Store in Redis under key img:json:x as a JSON object
        key = f"img:json:{i}"
        redis_client.json().set(key, "$", data)
        
    print(f"Stored {min(len(labels), len(images))} samples in Redis as JSON under img:json:x.")


def store_samples_in_redis_as_binary(redis_client, labels, images):
    """Store labels and binary image data in Redis separately.
    
    Args:
        redis_client: Redis connection object
        labels: List/array of labels
        images: List/array of image matrices in original format
    """
    for i, (label, image) in enumerate(zip(labels, images)):
        # Convert image to binary string format
        binary_string = serialize_to_binary(image)

        # Store in Redis under key img:label:{index} as an integer
        label_key = f"img:label:{i}"
        redis_client.set(label_key, int(label))
        
        # Store in Redis under key img:string:{index} as a binary string
        image_key = f"img:string:{i}"
        redis_client.set(image_key, binary_string)

    print(f"Stored {min(len(labels), len(images))} samples in Redis as img:label:x and img:string:x.")


if __name__ == "__main__":
    dataset_path = Path("dataset/gzip")

    # # Test set (10k images)   
    # images = read_idx_images(dataset_path / "emnist-mnist-test-images-idx3-ubyte.gz")
    # labels = read_idx_labels(dataset_path / "emnist-mnist-test-labels-idx1-ubyte.gz")
    # Training set (60k images)   
    images = read_idx_images(dataset_path / "emnist-mnist-train-images-idx3-ubyte.gz")
    labels = read_idx_labels(dataset_path / "emnist-mnist-train-labels-idx1-ubyte.gz")
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    print(f"Loaded {len(labels)} labels\n")
    
    # Display first few images
    for i in range(5):
        print(f"Image {i} - Label: {labels[i]}")
        display_image(images[i])

    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # Store just a few images
    store_samples_in_redis_as_binary(r, labels[0:200], images[0:200])
    store_samples_in_redis_as_json(r, labels[0:200], images[0:200])

    # # Store all images
    # store_samples_in_redis_as_binary(r, labels, images)
    # store_samples_in_redis_as_json(r, labels, images)