import gzip

import numpy as np


def open_file(path):
    """Open a file, handling gzip compression automatically."""
    path_str = str(path)
    return gzip.open(path, "rb") if path_str.endswith(".gz") else open(path, "rb")

def read_embeddings(path):
    with open_file(path) as f:
        embeddings = np.load(f)

    print(f"Read {len(embeddings)} with shape {embeddings.shape}.")
    return embeddings

