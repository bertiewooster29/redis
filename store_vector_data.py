import numpy as np
import redis
from pathlib import Path

# Load embeddings
embeddings_path = Path("dataset/embeddings")
print(f"Loading embeddings from {embeddings_path / 'mnist_embeddings.npy'}...")
embeddings = np.load(embeddings_path / 'mnist_embeddings.npy')
print(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Store first 200 embeddings
num_samples = 200
print(f"Storing first {num_samples} embeddings in Redis...")

for i in range(num_samples):
    key = f"img:vector:{i}"
    # Serialize embedding as bytes
    vector_bytes = embeddings[i].tobytes()
    r.set(key, vector_bytes)
    
    if (i + 1) % 50 == 0:
        print(f"  Stored {i + 1}/{num_samples} embeddings...")

print(f"Done! Stored {num_samples} embeddings in Redis")
