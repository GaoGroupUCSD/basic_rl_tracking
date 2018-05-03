import cv2
import numpy as np
from collections import defaultdict
from time import sleep
import os

files = os.listdir('horizontal_trajectories_random_direction/')
path = 'horizontal_trajectories_random_direction/'
trajectories = defaultdict(list)

for f in files:
    if not f.endswith('.png'):
        continue
    trajectories[f.split('_')[0]].append(path+f)

for t in trajectories:
    trajectories[t].sort(key = lambda x: int(x.split('_')[4].split('.')[0]))
    print t
    for i in trajectories[t]:
        img = cv2.imread(i)
        cv2.imshow('frame', img)
        cv2.waitKey(100)
    sleep(2)

# When everything done, release the capture
cv2.destroyAllWindows()

