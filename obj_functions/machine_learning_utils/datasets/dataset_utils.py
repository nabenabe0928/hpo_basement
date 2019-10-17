import torch
import numpy as np
from cifar import get_cifar
from svhn import get_svhn
from torch.utils.data.dataset import Subset
from imagenet import get_imagenet


def get_data(dataset_name,
             batch_size,
             n_cls=10,  # The number of class in training and testing
             image_size=None,  # pixel size
             sub_prop=None,  # How much percentages of training we use in an experiment. [0, 1]
             biased_cls=None  # len(biased_cls) must be n_cls. Each element represents the percentages.
             ):

    train_raw_dataset, test_raw_dataset, raw_n_cls = get_raw_dataset(dataset_name, batch_size, image_size, n_cls)

    n_cls = None if n_cls == raw_n_cls else n_cls
    train_dataset, test_dataset = process_raw_dataset(train_raw_dataset,
                                                      test_raw_dataset,
                                                      raw_n_cls,
                                                      n_cls,
                                                      sub_prop,
                                                      biased_cls)

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_data, test_data


def get_raw_dataset(dataset_name, batch_size, image_size=None, n_cls=10):
    if dataset_name.upper() == "CIFAR":
        if 2 <= n_cls <= 10:
            nc = 10
        elif 11 <= nc <= 100:
            nc = 100
        else:
            raise ValueError("n_cls must be between 2 and 100.")

        return get_cifar(batch_size, nc) if image_size is None else get_cifar(batch_size, nc, image_size)
    elif dataset_name.upper() == "SVHN":
        return get_svhn(batch_size) if image_size is None else get_svhn(batch_size, image_size)
    elif dataset_name.upper() == "IMAGENET":
        return get_imagenet(batch_size) if image_size is None else get_svhn(batch_size, image_size)


def process_raw_dataset(train_raw_dataset,
                        test_raw_dataset,
                        raw_n_cls,
                        n_cls=None,
                        sub_prop=None,
                        biased_cls=None):

    if n_cls is None and sub_prop is None and biased_cls is None:
        return train_raw_dataset, test_raw_dataset
    else:
        train_itr = range(len(train_raw_dataset))
        test_itr = range(len(train_raw_dataset))
        train_labels = [[train_raw_dataset[i][1] for i in train_itr],
                        list(train_itr)]
        test_labels = [[test_raw_dataset[i][1] for i in test_itr],
                       list(test_itr)]
        train_labels, test_labels = map(np.asarray, [train_labels, test_labels])

        if n_cls is not None:
            train_labels, test_labels = get_small_class(train_labels, test_labels, n_cls)
        if sub_prop is not None:
            n_subtrain = int(np.ceil(len(train_labels) * sub_prop))
            train_labels = np.array([tl[:n_subtrain] for tl in train_labels])
        if biased_cls is not None:
            train_labels = get_biased_class(train_labels, biased_cls, n_cls, raw_n_cls)

        train_indexes = np.array([datum[0] for datum in train_labels])
        test_indexes = np.array([datum[0] for datum in test_labels])

        return Subset(train_raw_dataset, train_indexes), Subset(test_raw_dataset, test_indexes)


def get_small_class(train_labels, test_labels, n_cls):
    train_indexes, test_indexes = [], []

    for idx, label in enumerate(train_labels[0]):
        if label < n_cls:
            train_indexes.append(idx)
    for idx, label in enumerate(test_labels[0]):
        if label < n_cls:
            test_indexes.append(idx)

    train_indexes, test_indexes = map(np.asarray, [train_indexes, test_indexes])

    return np.array([tl[train_indexes] for tl in train_labels]), np.array([tl[test_indexes] for tl in test_labels])


def get_biased_class(train_labels, biased_cls, n_cls, raw_n_cls):
    if len(biased_cls) != n_cls and len(biased_cls) != raw_n_cls:
        raise ValueError("The length of biased_cls must be n_cls(={}) or {}, but {} was given.".format(n_cls, raw_n_cls, len(biased_cls)))
    n_cls = raw_n_cls if n_cls is None else n_cls

    labels_to_idx = [train_labels[1][np.where(train_labels[0] == n)[0]] for n in range(len(n_cls))]
    labels_to_idx = [idx[:int(len(idx) * biased_cls[n])] for n, idx in enumerate(labels_to_idx)]

    return_idx = []
    for idx in labels_to_idx:
        return_idx += idx

    return_idx = np.array(return_idx)

    return np.array([np.sort(return_idx), list(len(return_idx))])
