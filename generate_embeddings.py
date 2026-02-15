## generate_embeddings.py

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gzip
import struct
from pathlib import Path

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


# Define transforms: Convert to tensor, repeat grayscale to 3 channels (for RGB pre-trained model),
# and normalize using ImageNet stats (standard for pre-trained ResNet).
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat single channel to 3
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load MNIST dataset from local gz files
dataset_path = Path("dataset/gzip")
embeddings_path = Path("dataset/embeddings")
embeddings_path.mkdir(parents=True, exist_ok=True)

print(f"Loading images from {dataset_path / 'emnist-mnist-train-images-idx3-ubyte.gz'}...")
images = read_idx_images(dataset_path / 'emnist-mnist-train-images-idx3-ubyte.gz')
print(f"Loaded {len(images)} images")

print(f"Loading labels from {dataset_path / 'emnist-mnist-train-labels-idx1-ubyte.gz'}...")
labels_array = read_idx_labels(dataset_path / 'emnist-mnist-train-labels-idx1-ubyte.gz')
print(f"Loaded {len(labels_array)} labels")

# Convert images to float32 and normalize to [0, 1]
images = images.astype(np.float32) / 255.0

# Apply transforms to each image
print("Applying transforms to images...")
transformed_images = []
for i, img in enumerate(images):
    # Convert numpy array to PIL Image for transforms
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
    img_tensor = img_tensor.repeat(3, 1, 1)  # Repeat to 3 channels
    # Normalize
    img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
    transformed_images.append(img_tensor)
    
    # Report progress every 2000 images
    if (i + 1) % 2000 == 0:
        print(f"  Transformed {i + 1}/{len(images)} images...")

print(f"Finished transforming all {len(images)} images")

# Stack all images into a single tensor
print("Creating dataset and dataloader...")
images_tensor = torch.stack(transformed_images)
labels_tensor = torch.from_numpy(labels_array.copy()).long()  # Make a writable copy

# Create dataset and dataloader
mnist_dataset = TensorDataset(images_tensor, labels_tensor)
data_loader = DataLoader(mnist_dataset, batch_size=512, shuffle=False)  # Large batch for efficiency

# Load pre-trained ResNet-18
print("Loading pre-trained ResNet-18 model...")
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # Set to evaluation mode

# Create feature extractor: Remove the final fully connected layer to get 512-dim embeddings
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Generate embeddings
print("Generating embeddings...")
embeddings = []
labels = []  # Optional: Store labels if needed

with torch.no_grad():  # No gradients for inference
    for batch_idx, (images, lbls) in enumerate(data_loader):
        # Extract features (output: batch_size x 512 x 1 x 1)
        features = feature_extractor(images)
        # Flatten to batch_size x 512
        features = features.view(features.size(0), -1)
        embeddings.append(features.cpu().numpy())  # Move to CPU and convert to NumPy
        labels.append(lbls.numpy())
        
        # Report progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            processed = (batch_idx + 1) * data_loader.batch_size
            total = len(mnist_dataset)
            print(f"  Processed {processed}/{total} images ({100 * processed / total:.1f}%)...")

# Concatenate all embeddings and labels
print("Concatenating embeddings and labels...")
embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

# Example: Print shape and a sample embedding
print(f"Embeddings shape: {embeddings.shape}")  # Should be (60000, 512)
print(f"Sample embedding for first image: {embeddings[0][:10]}...")  # Truncated for brevity

# Save to file
print(f"Saving embeddings to {embeddings_path / 'mnist_embeddings.npy'}...")
np.save(embeddings_path / 'mnist_embeddings.npy', embeddings)
print(f"Saving labels to {embeddings_path / 'mnist_labels.npy'}...")
np.save(embeddings_path / 'mnist_labels.npy', labels)
print("Done!")
