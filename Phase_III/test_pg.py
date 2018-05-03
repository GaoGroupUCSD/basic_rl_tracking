import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from env import *
from network import *
from actor import *
from critic import *
from general_functions import *
import sys
import time

LIMITS = (0, 9)
NUM_ACTIONS = 3
MAX_STEPS = 10

def test_pg(ckpt):

    print ckpt

    e = env()
    policy_network = torch.load(ckpt)

    num_steps = 0.0
    num_correct_steps = 0.0
    num_correct_episodes = 0.0

    for trajectory in xrange(1000):

        s = e.first_time_step()
        a_probs = policy_network(convert_to_variable(s))
        a = greedy(a_probs)
        #print a_probs
        #print 'action: ' + representation(a)
        e.perform_action(a)
        visualize_1D(e)
        if e.get_board() == e.get_box():
            num_correct_steps += 1
        num_steps += 1

        while not e.done():
            s = e.next_time_step()
            a_probs = policy_network(convert_to_variable(s))
            a = greedy(a_probs)
            #print a_probs
            #print 'action: ' + representation(a)
            e.perform_action(a)
            visualize_1D(e)
            if e.get_board() == e.get_box():
                num_correct_steps += 1
            num_steps += 1

        if e.get_board() == e.get_box():
            num_correct_episodes += 1

    return num_correct_steps/num_steps, num_correct_episodes/1000

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Enter checkpoint for policy network with path'
        sys.exit()
    
    test_pg(sys.argv[1])
