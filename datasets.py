"""Datasets module. Provides utilities to load popular datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import numpy as np
import os

data_dir = 'cifar-10-batches-py'


def load_cifar10_dataset():
    """Load the cifar10 dataset.

    :return: train, test data:
            for (X)
    """
    # d
    all_x = None

    # Training set
    training_x = None

    # Test set
    testing_x = np.array([])

    for fn in os.listdir(data_dir):

        if not fn.startswith('batches') and not fn.startswith('readme'):
            fo = open(os.path.join(data_dir, fn), 'rb')
            data_batch = pickle.load(fo)
            fo.close()

            if fn.startswith('data'):

                if training_x is None:
                    training_x = data_batch['data']
                else:
                    training_x = np.concatenate((training_x, data_batch['data']), axis=0)

                if all_x is None:
                    all_x = data_batch['data']
                else:
                    all_x = np.concatenate((all_x, data_batch['data']), axis=0)

            if fn.startswith('test'):
                testing_x = data_batch['data']

                if all_x is None:
                    all_x = data_batch['data']
                else:
                    all_x = np.concatenate((all_x, data_batch['data']), axis=0)

    all_x = all_x.astype(np.float32) / 255.0
    training_x = training_x.astype(np.float32) / 255.0
    testing_x = testing_x.astype(np.float32) / 255.0

    return all_x, training_x, testing_x

