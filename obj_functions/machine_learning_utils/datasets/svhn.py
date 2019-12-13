from torchvision import datasets, transforms
import numpy as np
import warnings


# https://github.com/Coderx7/SimpleNet_Pytorch/issues/3


def get_svhn(image_size=32, test=False, all_train=False):
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
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

    train_dataset = datasets.SVHN(root="svhn",
                                  split="train",
                                  download=True,
                                  transform=transform_train)

    n_all = len(train_dataset)
    n_train = int(n_all * 0.8) if not all_train else n_all
    train_labels = np.array([train_dataset.labels[:n_train], list(range(n_train))])

    if test:
        test_dataset = datasets.SVHN(root="svhn",
                                     split="test",
                                     download=True,
                                     transform=transform_test)
        test_labels = np.array([test_dataset.labels, list(range(len(test_dataset)))])
    else:
        test_dataset = datasets.SVHN(root="svhn",
                                     split="train",
                                     download=False,
                                     transform=transform_test)
        test_labels = np.array([test_dataset.labels[n_train:], list(range(n_train, n_all))])

    return train_dataset, train_labels, test_dataset, test_labels, 10
