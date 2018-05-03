import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

INPUT_SIZE = 20
NUM_ACTIONS = 3

class actor(nn.Module):
    def __init__(self):
        super(actor, self).__init__()
        self.step = 0
        self.hidden1 = nn.Linear(INPUT_SIZE, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 3)

    def forward(self, X):
        h1 = F.relu(self.hidden1(X))
        h2 = F.relu(self.hidden2(h1))
        o = F.softmax(self.output(h2))
        #return Variable(o.data.view(1, 3), requires_grad=True)
	return o.unsqueeze(0)
