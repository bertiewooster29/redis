import redis
import numpy as np

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


# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Create vector index
print("Creating vector index...")
try:
    r.ft("mnist_vec_idx").create_index([
        redis.commands.search.field.VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": 512,
                "DISTANCE_METRIC": "COSINE",
                "M": 16
            }
        )
    ], definition=redis.commands.search.IndexDefinition(prefix=["img:vector:"]))
    print("Index created successfully")
except redis.exceptions.ResponseError as e:
    if "Index already exists" in str(e):
        print("Index already exists, skipping creation")
    else:
        raise

# Get the vector at index 17
query_key = "img:vector:17"
query_vector_bytes = r.hget(query_key, "embedding")
query_label = r.hget(query_key, "label")

print(f"\nQuery vector: {query_key}")
print(f"Query label: {int(query_label)}")

# Get the original image for display
query_image_key = "img:hset:17"
query_image_bytes = r.hget(query_image_key, "pixels")
query_image = np.frombuffer(query_image_bytes, dtype=np.uint8).reshape(28, 28)

print("\nQuery image:")
display_image(query_image)

# Convert bytes to numpy array
query_vector = np.frombuffer(query_vector_bytes, dtype=np.float32)

# Search for 5 nearest neighbors (including itself)
print(f"Searching for 5 nearest neighbors...")
from redis.commands.search.query import Query

q = Query(f"*=>[KNN 5 @embedding $vec AS score]").return_fields("label", "score").sort_by("score").dialect(2)
results = r.ft("mnist_vec_idx").search(q, query_params={"vec": query_vector.tobytes()})

print(f"\nFound {results.total} results:")
for i, doc in enumerate(results.docs):
    # Extract index from key (e.g., "img:vector:17" -> 17)
    idx = int(doc.id.split(":")[-1])
    
    # Get the original image for this result
    image_key = f"img:hset:{idx}"
    image_bytes = r.hget(image_key, "pixels")
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(28, 28)
    
    print(f"\n{i+1}. Key: {doc.id}, Label: {doc.label.decode()}, Distance: {doc.score}")
    display_image(image)

