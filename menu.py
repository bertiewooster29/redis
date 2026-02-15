import redis

from idx_tools import read_emnist_data, display_image
from embedding_tools import read_embeddings
import store_image_data as sid

# Constants
LABELS_FILE = "dataset/gzip/emnist-mnist-train-labels-idx1-ubyte.gz"
IMAGE_FILE = "dataset/gzip/emnist-mnist-train-images-idx3-ubyte.gz"
EMBEDDINGS_FILE = "dataset/embeddings/mnist_embeddings.npy.gz"


def f_AAA():
    print("AAA function called")


def print_menu():
    print("\n=== Menu ===")
    print("1. Store raw data (separate labels, images, embeddings).")
    print("2. Store JSON data.")
    print("3. Store HASH data.")
    print("4. AAA")
    print("0. Exit")
    
    choice = input("\nEnter your choice: ")
    
    if choice == "1":
        sid.store_as_raw(r, labels, images)
    elif choice == "2":
        sid.store_as_json(r, labels, images, embeddings)
    elif choice == "3":
        sid.store_as_hash(r, labels, images, embeddings)
    elif choice == "4":
        f_AAA()
    elif choice == "0":
        print("Exiting...")
        return False
    else:
        print("Invalid choice. Please try again.")
    
    return True


if __name__ == "__main__":
    # Connect to Redis. decode_responses must be False. Otherwise, Redis will insist
    # on reading strings _as if_ they were encoded in UTF-8 even if they are binary
    r = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Load data from files
    labels, images = read_emnist_data(LABELS_FILE, IMAGE_FILE)
    embeddings = read_embeddings(EMBEDDINGS_FILE)

    while print_menu():
        pass
