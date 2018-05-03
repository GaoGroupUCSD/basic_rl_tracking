import numpy as np
import torch
import cv2
from torch.autograd import Variable

NUM_ACTIONS = 3
GAMMA = 0.9

def epsilon_greedy(q_values, epsilon):
    q_values = q_values.data.numpy()
    if np.random.rand() <= epsilon:
        return np.random.randint(0, NUM_ACTIONS)
    return np.argmax(q_values)

def sample(probs):
    x = np.random.rand()
    prob_sum = 0.0
    probs = probs.data.numpy()[0]
    for i in xrange(len(probs)):
        prob_sum += probs[i]
	if x <= prob_sum:
	    return i
    return len(probs) - 1

def compute_discounted_rewards(reward, actions):
    y = []
    running_reward = reward
    #distributed_reward = reward / len(q_values)
    for i in xrange(len(actions) - 1, -1, -1):
    	y = [running_reward] + y
    	running_reward = GAMMA * running_reward
    '''y.append([reward if x == actions[-1] else q_values[-1][x] for x in xrange(NUM_ACTIONS)])
    for i in xrange(len(q_values) - 2, -1, -1):
        y = [[q_values[i][x] if x != actions[i] else GAMMA * np.max(q_values[i + 1]) for x in xrange(NUM_ACTIONS)]] + y'''
    y = np.array(y)
    return convert_to_variable(y, False)

def convert_to_variable(x, grad=True):
    return Variable(torch.FloatTensor(x), requires_grad=grad)

def greedy(x):
    y = x.data.numpy()
    return np.argmax(y)

def representation(action):
    if action == 0:
        return '<-'
    elif action == 1:
        return '.'
    return '->'

def visualize_1D(e):
    board = e.get_board()
    box = e.get_box()
    #print board
    #print box
    #print
    # Creating image to display actions in 2-D
    img = np.zeros((100, 100, 3))
    # Multiplying board and box coordinates by 10 to make moves made more apparent and fixing object height to 10 (50 - 60)
    cv2.rectangle(img, (board * 10, 50), ((board + 1) * 10, 60), (255, 255, 255), -1)
    cv2.rectangle(img, (box * 10, 50), ((box + 1) * 10, 60), (0, 0, 255), 2)
    cv2.imshow('vis', img)
    cv2.waitKey(1)
    # If the episode has terminated, or our estimate has gone out of bounds, we close the window
    if e.done():
        cv2.destroyAllWindows()
    return

def compute_pg_gradients(actions, probs, discounted_rewards):
    loss = []
    y = []
    for i in xrange(len(actions)):
    	y.append([probs.data[i][x] for x in xrange(NUM_ACTIONS)])
	y[i][actions[i]] = 1
    y = convert_to_variable(y, False)
    '''print 'y'
    print y.data
    print 'probs'
    print probs.data
    print 'discounted rewards' 
    print discounted_rewards.data'''
    # NOTE: Update rule to compute actor gradient
    # theta <- theta + step_size * (action_taken - prob) * discounted_reward
    #for i in xrange(len(actions)):
    #    loss.append((y.data[i] - probs.data[i]) * discounted_rewards.data[i])
    gradients = ((y - probs).t() * discounted_rewards).t()
    #print loss
    #loss = loss.sum(dim=0).sum(dim=0)
    #print torch.mean(loss, dim=0)
    return gradients

def compute_critic_gradients(q_values, target_values):
    q_values = q_values.data.numpy()
    target_values = target_values.data.numpy()
    # NOTE: Here, we return -(target-q_values) because in the update equation, we have:
    # w <- w + lr * TDerror
    # Since optimizers generally do w <- w - lr * error, we pass -TDerror as the gradients.
    return convert_to_variable(q_values - target_values)

def print_gradients(net):
    for p in net.parameters():
        print 'Data: ' + str(p.data)
        print 'Gradient: ' + str(p.grad)

