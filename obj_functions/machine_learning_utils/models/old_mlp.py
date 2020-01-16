import torch.nn as nn
import torch.nn.functional as F


class OldMultiLayerPerceptron(nn.Module):
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
    image_size: int
        The size of column or row of training image.
    drop_rate1, 2: float
        The probability of dropout a weight of connections.
    n_units1, 2: int
        the number of hidden layer units
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
                 lr=5.0e-2,
                 momentum=0.9,
                 weight_decay=5.0e-4,
                 image_size=28,
                 n_units=275,
                 drop_rate=0.4,
                 nesterov=True,
                 lr_decay=1.,
                 lr_step=[1.],
                 n_cls=10
                 ):
        super(OldMultiLayerPerceptron, self).__init__()

        # Hyperparameter Configuration for MLP.
        self.batch_size = int(batch_size)
        self.lr = lr
        self.momentum = momentum
        self.image_size = image_size if image_size is not None else 28
        self.weight_decay = weight_decay
        self.drop_rate1 = drop_rate
        self.drop_rate2 = drop_rate
        self.n_units1 = int(n_units)
        self.n_units2 = int(n_units)
        self.nesterov = nesterov
        self.lr_decay = lr_decay
        self.epochs = 20
        self.lr_step = [int(step * self.epochs) for step in lr_step]

        # Architecture of MLP.
        self.full_conn1 = nn.Linear(self.image_size * self.image_size, self.n_units1)
        self.full_conn2 = nn.Linear(self.n_units1, self.n_units2)
        self.full_conn3 = nn.Linear(self.n_units2, n_cls)
        self.init_inner_params()

    def init_inner_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1, self.image_size * self.image_size)
        h = self.full_conn1(x)
        h = F.relu(h)
        h = F.dropout2d(h, p=self.drop_rate1, training=self.training)
        h = self.full_conn2(h)
        h = F.relu(h)
        h = F.dropout2d(h, p=self.drop_rate2, training=self.training)
        h = self.full_conn3(h)

        return F.log_softmax(h, dim=1)
