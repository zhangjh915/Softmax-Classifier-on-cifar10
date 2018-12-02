import _pickle as pickle
import numpy as np
import os


def load_data_batch(filename):
    """ Load one batch of cifar in 'cifar-10-batches-py' with name 'data_batch_i'."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        x, y = data['data'], np.array(data['labels'])
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        return x, y


def load_data(path):
    """ Load all batches of cifar in 'cifar-10-batches-py' with name 'data_batch_i' as training sets
        and 'test_batch' as test set."""
    xtrain = []
    ytrain = []
    for i in range(1, 6):
        f = os.path.join(path, 'data_batch_%d' % i)
        x, y = load_data_batch(f)
        xtrain.append(x)
        ytrain.append(y)
    xtrain = np.concatenate(xtrain)
    ytrain = np.concatenate(ytrain)
    xtest, ytest = load_data_batch(os.path.join(path, 'test_batch'))
    return xtrain, ytrain, xtest, ytest


def load_dataset():
    # Load data-set.
    data_path = 'data/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_data(data_path)

    # Pre-process. Reshape RGB images into unbiased row vectors.
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_mean_train = np.mean(x_train, axis=0)
    x_train -= x_mean_train
    x_test -= x_mean_train

    # Add a row in data to fit the relation y = Wx.
    x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))]).T
    x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))]).T
    return x_train, y_train, x_test, y_test
