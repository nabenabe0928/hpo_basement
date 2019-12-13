from torchvision import datasets, transforms
import numpy as np


# https://gluon-cv.mxnet.io/build/examples_classification/demo_cifar10.html
# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151


def get_cifar(n_cls=10, image_size=32, test=False, all_train=False):
    if n_cls == 10:
        return get_cifar10(image_size, test=test, all_train=all_train)
    elif n_cls == 100:
        return get_cifar100(image_size, test=test, all_train=all_train)
    else:
        raise ValueError("The number of class for CIFAR must be 2 to 100.")
        print("But, {} was given.".format(n_cls))


def get_cifar10(image_size=32, test=False, all_train=False):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])

    transform_train = transforms.Compose([transforms.Pad(4, padding_mode='reflect'),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(image_size),
                                          transforms.ToTensor(),
                                          normalize])
    transform_test = transforms.Compose([transforms.CenterCrop(image_size),
                                         transforms.ToTensor(),
                                         normalize])

    train_dataset = datasets.CIFAR10(root="cifar",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
    n_all = len(train_dataset)
    n_train = int(n_all * 0.8) if not all_train else n_all
    train_labels = np.array([train_dataset.targets[:n_train], list(range(n_train))])

    if test:
        test_dataset = datasets.CIFAR10(root="cifar",
                                        train=False,
                                        download=False,
                                        transform=transform_test)
        test_labels = np.array([test_dataset.targets, list(range(len(test_dataset)))])
    else:
        test_dataset = datasets.CIFAR10(root="cifar",
                                        train=True,
                                        download=False,
                                        transform=transform_test)
        test_labels = np.array([test_dataset.targets[n_train:], list(range(n_train, n_all))])

    return train_dataset, train_labels, test_dataset, test_labels, 10


def get_cifar100(image_size=32, test=False, all_train=False):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    transform_train = transforms.Compose([transforms.Pad(4, padding_mode='reflect'),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(image_size),
                                          transforms.ToTensor(),
                                          normalize])
    transform_test = transforms.Compose([transforms.CenterCrop(image_size),
                                         transforms.ToTensor(),
                                         normalize])

    train_dataset = datasets.CIFAR100(root="cifar",
                                      train=True,
                                      download=True,
                                      transform=transform_train)

    n_all = len(train_dataset)
    n_train = int(n_all * 0.8) if not all_train else n_all
    train_labels = np.array([train_dataset.targets[:n_train], list(range(n_train))])

    if test:
        test_dataset = datasets.CIFAR100(root="cifar",
                                         train=False,
                                         download=False,
                                         transform=transform_test)
        test_labels = np.array([test_dataset.targets, list(range(len(test_dataset)))])
    else:
        test_dataset = datasets.CIFAR100(root="cifar",
                                         train=True,
                                         download=False,
                                         transform=transform_test)
        test_labels = np.array([test_dataset.targets[n_train:], list(range(n_train, n_all))])

    return train_dataset, train_labels, test_dataset, test_labels, 100
