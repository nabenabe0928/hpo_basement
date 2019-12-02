import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


"""
reference: http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
Densely Connected Convolutional Networks
Gao Huang et al.

bibtex
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}
"""


class TransionLayer(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.2):
        super(TransionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        h = F.relu(self.bn(x), inplace=True)
        h = self.conv(h)
        h = F.dropout(h, p=self.drop_rate, training=self.training)
        h = F.avg_pool2d(h, 2)
        return h


class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth_rate, growth_coef=4, drop_rates=[0.2, 0.2]):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(growth_rate * growth_coef)
        self.conv1 = nn.Conv2d(in_ch, growth_rate * growth_coef, 1, padding=0, bias=False),
        self.conv2 = nn.Conv2d(growth_rate * growth_coef, growth_rate, 3, padding=1, bias=False)
        self.drop_rates = drop_rates

    def forward(self, x):
        # Reduce the image size (bottleneck layer)
        h = F.relu(self.bn1(x), inplace=True)
        h = self.conv1(h)
        h = F.dropout(h, p=self.drop_rates[0], training=self.training)

        # Feature extractor
        h = F.relu(self.bn2(h), inplace=True)
        h = self.conv2(h)
        h = F.dropout(h, p=self.drop_rates[1], training=self.training)

        return torch.cat((h, x), dim=1)


class DenseNetBC(nn.Module):
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
    n_layers1 ,2 ,3 : int
        The number of repeating the dense block
    compression1 ,2 : float
        The ratio of compression of the input size from the origin to the next layer
    growth_rate1 ,2 ,3 : int
        The number of feature maps in the dense block
    growth_coef1 ,2 ,3, 4 : int
        The coefficient for the number of feature maps from the bottleneck layer.
    drop_1st_db1, 2, 3: float
        The probability of dropout a weight of connections in the first conv of dense block 1, 2, 3
    drop_2nd_db1, 2, 3: float
        The probability of dropout a weight of connections in the second conv of dense block 1, 2, 3
    drop_tl1, 2: float
        The probability of dropout a weight of connections in a transition layer 1, 2
    nesterov: bool
        Whether using nesterov or not in SGD.
    epochs: int
        The number of training throughout one learning process.
    lr_step: list of float
        When to decrease the learning rate.
        The learning rate will decline at lr_step[k] * epochs epoch.
    lr_decay: float
        How much make learning rate decline at epochs * lr_step[k]
    n_cls: int
        The number of classes on a given task.
    """

    def __init__(self,
                 batch_size=64,
                 lr=0.1,
                 momentum=0.9,
                 weight_decay=1.0e-4,
                 n_layers1=16, n_layers2=16, n_layers3=16,
                 compression1=0.5, compression2=0.5,
                 growth_rate1=24, growth_rate2=24, growth_rate3=24,
                 growth_coef1=2, growth_coef2=4, growth_coef3=4, growth_coef4=4,
                 drop_1st_db1=0.2, drop_2nd_db1=0.2,
                 drop_1st_db2=0.2, drop_2nd_db2=0.2,
                 drop_1st_db3=0.2, drop_2nd_db3=0.2,
                 drop_tl1=0.2, drop_tl2=0.2,
                 nesterov=True,
                 epochs=300,
                 lr_step=[0.5, 0.75, 1],
                 lr_decay=0.1,
                 n_cls=100
                 ):
        super(DenseNetBC, self).__init__()

        n_layers = [int(n_layers1), int(n_layers2), int(n_layers3)]
        drop_db = [[drop_1st_db1, drop_2nd_db1], [drop_1st_db2, drop_2nd_db2], [drop_1st_db3, drop_2nd_db3]]
        drop_tl = [drop_tl1, drop_tl2]
        compressions = [compression1, compression2]
        growth_coefs = [int(growth_coef1), int(growth_coef2), int(growth_coef3), int(growth_coef4)]
        growth_rates = [int(growth_rate1), int(growth_rate2), int(growth_rate3)]

        # Hyperparameter Configuration for CNN.
        self.db = []
        self.tl = []
        self.lr = lr
        self.momentum = momentum
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr_decay = lr_decay
        self.lr_step = [int(step * self.epochs) for step in lr_step]
        self.n_cls = int(n_cls)
        self.weight_decay = weight_decay
        self.nesterov = bool(nesterov)

        # Architecture of CNN.
        in_ch = growth_coefs[0] * growth_rates[0]
        self.conv = nn.Conv2d(3, growth_coefs[0] * growth_rates[0], 3, padding=1, bias=False)

        for i in range(3):
            in_ch = int(np.floor(in_ch * compressions[i - 1])) if i != 0 else in_ch
            self.db.append(self._add_DenseBlock(n_layers[i], in_ch, drop_db[i], growth_rates[i], growth_coefs[i + 1]))
            in_ch += growth_rates[i] * n_layers[i]
            if i < 2:
                self.tl.append(TransionLayer(in_ch, int(np.floor(in_ch * compressions[i])), drop_tl[i]))

        self.bn = nn.BatchNorm2d(in_ch)
        self.full_conn = nn.Linear(in_ch, n_cls)
        self.init_inner_params()

    def forward(self, x):
        h = self.conv(x)

        for i in range(3):
            h = self.db[i](h)
            h = self.tl[i](h) if i < 2 else h

        h = F.relu(self.bn(h), inplace=True)
        h = F.avg_pool2d(h, h.size(2))
        h = h.view(h.size(0), -1)
        h = self.full_conn(h)

        return F.log_softmax(h, dim=1)

    def _add_DenseBlock(self, n_layers, in_ch, drop_rates, growth_rate, growth_coef):
        layers = []

        for i in range(int(n_layers)):
            layers.append(DenseBlock(in_ch + growth_rate * i, growth_rate, growth_coef=growth_coef, drop_rates=drop_rates))

        return nn.Sequential(*layers)

    def init_inner_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
