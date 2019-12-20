from torchvision import datasets, transforms
import numpy as np


def get_mnist(image_size=28, test=False, all_train=False):
    transform_train = transforms.Compose([transforms.RandomCrop(image_size),
                                          transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.CenterCrop(image_size),
                                         transforms.ToTensor()])

    train_dataset = datasets.MNIST(root="mnist",
                                   train=True,
                                   download=True,
                                   transform=transform_train)

    n_all = len(train_dataset)
    n_train = int(n_all * 0.8) if not all_train else n_all
    train_labels = np.array([train_dataset.targets[:n_train], list(range(n_train))])

    if test:
        test_dataset = datasets.MNIST(root="mnist",
                                      train=False,
                                      download=True,
                                      transform=transform_test)
        test_labels = np.array([test_dataset.targets, list(range(len(test_dataset)))])
    else:
        test_dataset = datasets.MNIST(root="mnist",
                                      train=True,
                                      download=False,
                                      transform=transform_test)
        test_labels = np.array([test_dataset.targets[n_train:], list(range(n_train, n_all))])

    return train_dataset, train_labels, test_dataset, test_labels, 10
