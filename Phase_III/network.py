import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

INPUT_SIZE = 20
NUM_ACTIONS = 3

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.step = 0
        self.hidden1 = nn.Linear(INPUT_SIZE, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 3)
	nn.init.normal(self.hidden1.weight, 0.3, 0.1)
	nn.init.constant(self.hidden1.bias, 0.2)
	nn.init.normal(self.hidden2.weight, 0.3, 0.1)
	nn.init.constant(self.hidden2.bias, 0.2)
	nn.init.normal(self.output.weight, 0.3, 0.1)
	nn.init.constant(self.output.bias, 0.2)

    def forward(self, X):
        o = F.relu(self.hidden1(X))
        o = F.relu(self.hidden2(o))
        o = F.leaky_relu(self.output(o))
        #return Variable(o.data.view(1, 3), requires_grad=True)
	return o
