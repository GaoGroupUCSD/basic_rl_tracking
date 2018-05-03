import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from env import *
from network import *
from general_functions import *

LIMITS = (0, 9)
NUM_ACTIONS = 3
MAX_STEPS = 10

if __name__ == '__main__':
    e = env()
    n = Network()
    loss = nn.MSELoss()
    optimizer = optim.SGD(n.parameters(), lr=0.01, momentum=0.9)
    epsilon = 0.2

    for trajectory in xrange(100000000):

        actions = []
        optimizer.zero_grad()

        s = e.first_time_step()
        q_values = n(convert_to_variable(s))
        a = epsilon_greedy(q_values, epsilon)
        next_s = e.perform_action(a)
        next_q_values = n(convert_to_variable(next_s))

        q_states = q_values
        q_next_states = next_q_values
        actions.append(a)

        while not e.done():
            s = e.next_time_step()
            q_values = n(convert_to_variable(s))
            a = epsilon_greedy(q_values, epsilon)
            next_s = e.perform_action(a)
            next_q_values = n(convert_to_variable(next_s))

            q_states = torch.cat((q_states, q_values), 0)
            q_next_states = torch.cat((q_next_states, next_q_values), 0)
            actions.append(a)
            
        r = e.get_reward() * 1.0

        # Distribute rewards and only consider actions that were taken
        y = calculate_next_returns(r, q_states, q_next_states, actions)

        # Training
        output = loss(q_states, y)
        output.backward()
        optimizer.step()

        epsilon = 1 / (5 + 0.01 * trajectory)

        if (trajectory + 1) % 5000 == 0:
            torch.save(n, 'checkpoints/n_' + str(trajectory + 1) + '.pkl')
            print str(trajectory + 1) + ' : loss = ' + str(output.data[0])
