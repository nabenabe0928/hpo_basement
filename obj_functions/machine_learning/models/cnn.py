import torch.nn as nn
import torch.nn.functional as F
import math


class CNN(nn.Module):
    """
    Parameters
    ----------
    batch_size: int
        batch size of image dataset
    lr: float
        The learning rate of inner weight parameter of CNN.
    momentum: float
        momentum coefficient for Stochastic Gradient Descent (SGD)
    weight_decay: float
        the coefficients of a regularization term for cross entropy
    drop_rate: float
        The probability of dropout a weight of connections.
    ch1, ch2, ch3, ch4: int
        the number of kernel feature maps
    nesterov: bool
        Whether using nesterov or not in SGD.
    epochs: int
        The number of training throughout one learning process.
    lr_step: list of float
        When to decrease the learning rate.
        The learning rate will decline at lr_step[k] * epochs epoch.
    lr_decay: float
        How much make learning rate decline at epochs * lr_step[k]
    """

    def __init__(self, hp_dict):
        super(CNN, self).__init__()
        self.batch_size = hp_dict["batch_size"]
        self.lr = hp_dict["lr"]
        self.momentum = hp_dict["momentum"]
        self.weight_decay = hp_dict["weight_decay"]
        self.drop_rate = hp_dict["drop_rate"]
        self.ch1 = int(hp_dict["ch1"])
        self.ch2 = int(hp_dict["ch2"])
        self.ch3 = int(hp_dict["ch3"])
        self.ch4 = int(hp_dict["ch4"])
        self.nesterov = False
        self.epochs = 160
        self.lr_decay = 1
        self.lr_step = [1]

        self.c1 = nn.Conv2d(3, self.ch1, 5, padding=2)
        self.c2 = nn.Conv2d(self.ch1, self.ch2, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(self.ch2)
        self.c3 = nn.Conv2d(self.ch2, self.ch3, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(self.ch3)
        self.full_conn1 = nn.Linear(self.ch3 * 3 ** 2, self.ch4)
        self.full_conn2 = nn.Linear(self.ch4, 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c1(x)
        h = F.relu(F.max_pool2d(h, 3, stride=2))
        h = self.c2(h)
        h = F.relu(h)
        h = self.bn2(F.avg_pool2d(h, 3, stride=2))
        h = self.c3(h)
        h = F.relu(h)
        h = self.bn3(F.avg_pool2d(h, 3, stride=2))

        h = h.view(h.size(0), -1)
        h = self.full_conn1(h)
        h = F.dropout2d(h, p=self.drop_rate, training=self.training)
        h = self.full_conn2(h)
        return F.log_softmax(h, dim=1)
