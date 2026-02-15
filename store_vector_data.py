import numpy as np
import redis
from pathlib import Path

# Load embeddings and labels
embeddings_path = Path("dataset/embeddings")
print(f"Loading embeddings from {embeddings_path / 'mnist_embeddings.npy'}...")
embeddings = np.load(embeddings_path / 'mnist_embeddings.npy')
print(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")

print(f"Loading labels from {embeddings_path / 'mnist_labels.npy'}...")
labels = np.load(embeddings_path / 'mnist_labels.npy')
print(f"Loaded {len(labels)} labels")

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Store first 200 embeddings with labels as HASHes
num_samples = 200
print(f"Storing first {num_samples} embeddings and labels in Redis as HASHes...")

for i in range(num_samples):
    key = f"img:vector:{i}"
    # Serialize embedding as bytes
    vector_bytes = embeddings[i].tobytes()
    
    # Store as HASH with label and embedding fields
    r.hset(key, mapping={
        "label": int(labels[i]),
        "embedding": vector_bytes
    })
    
    if (i + 1) % 50 == 0:
        print(f"  Stored {i + 1}/{num_samples} embeddings...")

print(f"Done! Stored {num_samples} embeddings with labels in Redis as HASHes")

