# The Pickle files below store 200 samples:
#  - images in the "original" format (3D matrix: numpy.ndarray of numpy.ndarray of numpy.ndarray of numpy.uint8)
#  - images in the binary format (list of bytes)
#  - images in the CSV format (list of str)
#  - labels in the original format (numpy.ndarray of numpy.uint8)

import pickle

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
