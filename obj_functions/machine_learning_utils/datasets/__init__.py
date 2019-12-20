from obj_functions.machine_learning_utils.datasets.dataset_utils import get_data, get_dataset
from obj_functions.machine_learning_utils.datasets.cifar import get_cifar
from obj_functions.machine_learning_utils.datasets.svhn import get_svhn
from obj_functions.machine_learning_utils.datasets.imagenet import get_imagenet
from obj_functions.machine_learning_utils.datasets.mnist import get_mnist
from obj_functions.machine_learning_utils.datasets.fashion_mnist import get_fashionmnist


__all__ = ["get_data",
           "get_cifar",
           "get_svhn",
           "get_mnist",
           "get_fashionmnist",
           "get_imagenet",
           "get_dataset"]
