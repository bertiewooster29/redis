# The Pickle files below store the first 200 samples in the training set:
#  - images in the "original" format (3D matrix: numpy.ndarray of numpy.ndarray of numpy.ndarray of numpy.uint8)
#  - images in the binary format (list of bytes)
#  - images in the CSV format (list of str)
#  - labels in the original format (numpy.ndarray of numpy.uint8)

import pickle
import redis

if __name__ == "__main__":

    with open('my_images.pkl', 'rb') as inf:
        my_images = pickle.load(inf)
        print(my_images)

    with open('my_images_binary.pkl', 'rb') as inf:
        my_images_binary = pickle.load(inf)
        print(my_images_binary)

    with open('my_images_csv.pkl', 'rb') as inf:
        my_images_csv = pickle.load(inf)
        print(my_images_csv)

    with open('my_labels.pkl', 'rb') as inf:
        my_labels = pickle.load(inf)
        print(my_labels)

    r = redis.Redis(host='localhost', port=6379, decode_responses=True)


def store_samples_in_redis(redis_client, labels, images_csv):
    """Store label and image data in Redis as JSON objects."""
    import json
    
    for i, (label, csv_string) in enumerate(zip(labels, images_csv)):
        # Convert CSV string to list of integers
        pixels = [int(val) for val in csv_string.split(",")]
        
        # Create JSON object
        data = {
            "label": int(label),
            "pixels": pixels
        }
        
        # Store in Redis with key pattern "sample:{index}"
        key = f"sample:{i}"
        redis_client.json().set(key, "$", data)
        
    print(f"Stored {len(labels)} samples in Redis")


def store_samples_in_redis_binary(redis_client, labels, images_binary):
    """Store labels and binary image data in Redis separately."""
    for i, (label, binary_data) in enumerate(zip(labels, images_binary)):
        # Store label as integer
        label_key = f"samples:label:{i}"
        redis_client.set(label_key, int(label))
        
        # Store binary image data
        image_key = f"samples:imgbin:{i}"
        redis_client.set(image_key, binary_data)
    
    print(f"Stored {len(labels)} labels and binary images in Redis")

