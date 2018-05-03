import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from env import *
from network import *
from general_functions import *
import sys

LIMITS = (0, 9)
NUM_ACTIONS = 3
MAX_STEPS = 10

if __name__ == '__main__':
    
    print sys.argv
    if len(sys.argv) < 2:
        print 'Enter checkpoint with path'
        sys.exit()

    e = env()
    net = torch.load(sys.argv[1])

    for trajectory in xrange(10):

        s = e.first_time_step()
        q_values = net(convert_to_variable(s))
	print q_values
        a = greedy(q_values)
        next_s = e.perform_action(a)
        #visualize_1D(e, next_s)

        while not e.done():
            s = e.next_time_step()
            q_values = net(convert_to_variable(s))
	    print q_values
            a = greedy(q_values)
            next_s = e.perform_action(a)
            #visualize_1D(e, next_s)
