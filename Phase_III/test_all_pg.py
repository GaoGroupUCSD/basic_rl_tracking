import os
import matplotlib.pyplot as plt
from test_pg import *

path = 'pg_checkpoints/'
ckpts = os.listdir(path)

os.system('rm step_results.txt')
os.system('rm episode_results.txt')
os.system('touch step_results.txt')
os.system('touch episode_results.txt')

for c in sorted(ckpts):
    step_accuracy, episode_accuracy = test_pg(path + c)
    
    with open('step_results.txt', 'a') as f:
        f.write(c + ':' + str(step_accuracy) + '\n')

    with open('episode_results.txt', 'a') as f:
        f.write(c + ':' + str(episode_accuracy) + '\n')

p = []

def generate_accuracy_plot(filename):
    accuracies = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            l = line.strip().split(':')
            accuracies[int(l[0].split('.')[0])] = float(l[1])
    x = sorted(accuracies.keys())
    y = [accuracies[i] for i in x]
    plt.ylim((0, 1.0))
    t, = plt.plot(x, y, label=filename.split('.')[0])
    p.append(t)

generate_accuracy_plot('step_results.txt')
generate_accuracy_plot('episode_results.txt')
plt.legend(handles=p)
plt.savefig('accuracies.png')
