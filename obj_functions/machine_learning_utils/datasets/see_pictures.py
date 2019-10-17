import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


"""
dataset: 5-d array
1d: the index of image (default: < 50000)
2d: pixel information (0) or label (1)
3d: the index of RGB (0, 1, 2)
4d: the index of a row vector of pixel information (default: < 32)
5d: the index of the i-th row vector (default: < 32)
"""


transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root="cifar",
                                 train=True,
                                 download=True,
                                 transform=transform_train)
test_dataset = datasets.CIFAR10(root="cifar",
                                train=False,
                                download=False,
                                transform=transform_test)


def see_cifar10(dataset, idx=0):
    classes = ('plane', 'car', 'bird',
               'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck')

    npimg = dataset[idx][0].numpy()
    print(classes[dataset[idx][1]])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def see_cifar100(dataset, idx=0):
    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
               'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
               'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
               'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
               'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
               'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
               'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
               'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
               'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
               'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
               'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
    npimg = dataset[idx][0].numpy()
    print(classes[dataset[idx][1]])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
