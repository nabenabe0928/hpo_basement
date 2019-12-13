import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
reference: https://arxiv.org/pdf/1605.07146.pdf
Wide Residual Networks
Zagoruyko, Sergey et al.

bibtex
@article{zagoruyko2016wide,
  title={Wide residual networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  journal={arXiv preprint arXiv:1605.07146},
  year={2016}
}
"""


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_rate=0.3, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.in_is_out = (in_ch == out_ch and stride == 1)
        self.drop_rate = drop_rate

        self.shortcut = nn.Sequential() if self.in_is_out else nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=1, bias=False)

    def forward(self, x):
        h = F.relu(self.bn1(x), inplace=True)
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=True)
        h = F.dropout(h, p=self.drop_rate, training=self.training)
        h = self.conv2(h)

        return h + self.shortcut(x)


class WideResNet(nn.Module):
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
    n_blocks1 ,2 ,3 : int
        The number of repeating the residual groups in a bottleneck block
    drop_rate1, 2, 3: float
        The probability of dropout a weight of connections in a bottleneck block 1, 2, 3
    width_coef1, 2, 3: int
        The factor of widening the number of kernel feature maps of each residual link in a bottleneck block 1, 2, 3
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
                 batch_size=128,
                 lr=1.0e-1,
                 momentum=0.9,
                 weight_decay=5.0e-4,
                 n_blocks1=4, n_blocks2=4, n_blocks3=4,
                 width_coef1=10, width_coef2=10, width_coef3=10,
                 drop_rate1=0.3, drop_rate2=0.3, drop_rate3=0.3,
                 nesterov=True,
                 lr_decay=0.2,
                 lr_step=[0.3, 0.6, 0.8],
                 n_cls=100
                 ):
        super(WideResNet, self).__init__()

        # Hyperparameter Configuration for CNN.
        self.batch_size = int(batch_size)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_blocks = [n_blocks1, n_blocks2, n_blocks3]
        self.n_chs = [16, 16 * width_coef1, 32 * width_coef2, 64 * width_coef3]
        self.epochs = 200
        self.lr_step = [int(step * self.epochs) for step in lr_step]
        self.nesterov = nesterov
        self.lr_decay = lr_decay

        # Architecture of CNN.
        self.conv1 = nn.Conv2d(3, self.n_chs[0], 3, padding=1, bias=False)
        self.conv2 = self._add_groups(n_blocks1, self.n_chs[0], self.n_chs[1], drop_rate1)
        self.conv3 = self._add_groups(n_blocks2, self.n_chs[1], self.n_chs[2], drop_rate2, stride=2)
        self.conv4 = self._add_groups(n_blocks3, self.n_chs[2], self.n_chs[3], drop_rate3, stride=2)

        self.bn = nn.BatchNorm2d(self.n_chs[3])
        self.full_conn = nn.Linear(self.n_chs[3], n_cls)
        self.init_inner_params()

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

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)

        h = F.relu(self.bn(h), inplace=True)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.n_chs[3])
        h = self.full_conn(h)

        return F.log_softmax(h, dim=1)

    def _add_groups(self, n_blocks, in_ch, out_ch, drop_rate, stride=1):
        blocks = []

        for _ in range(int(n_blocks)):
            blocks.append(BasicBlock(in_ch, out_ch, stride=stride, drop_rate=drop_rate))
            in_ch, stride = out_ch, 1

        return nn.Sequential(*blocks)
