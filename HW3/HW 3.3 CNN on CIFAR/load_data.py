import numpy as np
import os

try:
    from six.moves import cPickle as pickle
except:
    import pickle


def unpickle(file_name):
    with open(file_name, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
        return dict


########################################################################
# Various constants for the size of the images.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels


########################################################################

def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float)  # / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def load_cifar10(dir):
    data_batches_names = ['data_batch_{}'.format(i) for i in range(1, 6)]
    X_train, y_train = [], []

    # loading training data and labels
    for batch_name in data_batches_names:
        data_dict = unpickle(os.path.join(dir, batch_name))
        X_train.append(data_dict['data'])
        y_train.append(data_dict['labels'])

    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    X_train = X_train.astype('float')

    # loading test data and labels
    data_dict = unpickle(os.path.join(dir, 'test_batch'))
    X_test, y_test = data_dict['data'], np.array(data_dict['labels'])
    X_test = X_test.astype('float')

    # Convert the images.
    X_train = _convert_images(X_train)
    X_test = _convert_images(X_test)

    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))
    return X_train, y_train, X_test, y_test


def load_cifar100(dir):
    data_dict = unpickle(os.path.join(dir, 'train'))
    X_train, y_train = data_dict['data'], np.array(data_dict['fine_labels'])
    X_train = X_train.astype('float')

    # loading test data and labels
    data_dict = unpickle(os.path.join(dir, 'test'))
    X_test, y_test = data_dict['data'], np.array(data_dict['fine_labels'])
    X_test = X_test.astype('float')

    # Convert the images.
    X_train = _convert_images(X_train)
    X_test = _convert_images(X_test)

    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))
    return X_train, y_train, X_test, y_test
