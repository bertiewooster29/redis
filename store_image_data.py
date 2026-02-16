import redis
import numpy as np

def store_as_raw(r, labels, images):
    """Store labels and binary image data in Redis separately.
    
    Args:
        r: Redis connection object
        labels: Array of labels
        images: Array of image matrices (28x28 unit8)
    """
    for i, (label, image) in enumerate(zip(labels, images)):
        # Store in Redis under key img:label:{index} as an integer
        label_key = f"img:label:{i}"
        r.set(label_key, int(label))
        
        # Store in Redis under key img:image:{index} as a binary string
        image_key = f"img:raw_image:{i}"
        r.set(image_key, image.tobytes())

    print(f"Stored {min(len(labels), len(images))} samples in Redis as img:label:<n> and img:raw_image:<n>.")

def store_as_json(r, labels, images, embeddings):
    """Store label, image, and embedding data in Redis as JSON.
    
    Args:
        redis_client: Redis connection object
        labels: Array of labels
        images: Array of image matrices (28x28 unit8 each)
        embeddings: Array of embeddings (512 floats each)
    """
    for i, (label, image, embedding) in enumerate(zip(labels, images, embeddings)):
        # Convert image to array of integers
        pixel_list = image.flatten().tolist()

        # Convert embedding to array of floats
        embedding_list = embedding.tolist()

        # Create JSON object
        data = {
            "label": int(label),
            "pixels": pixel_list,
            "embedding": embedding_list
        }
        # Store in Redis under key img:json:x as a JSON object
        key = f"img:json:{i}"
        r.json().set(key, "$", data)
        
    print(f"Stored {min(len(labels), len(images), len(embeddings))} samples in Redis as JSON under img:json:<n>.")



def store_as_hash(r, labels, images, embeddings):
    """Store label, image, and embedding data in Redis as HASHes.
    
    Args:
        redis_client: Redis connection object
        labels: Array of labels
        images: Array of image matrices (28x28 unit8 each)
        embeddings: Array of embeddings (512 floats each)
    """
    for i, (label, image, embedding) in enumerate(zip(labels, images, embeddings)):
        # Create mapping
        data = {
            "label": int(label),
            "pixels": image.flatten().tobytes(),
            "embedding": embedding.tobytes()
        }
        
        # Store in Redis under key img:hash:<n> as a HASH
        key = f"img:hash:{i}"
        r.hset(key, mapping=data)
        
    print(f"Stored {min(len(labels), len(images), len(embeddings))} samples in Redis as HSETs under img:hash:<n>.")
