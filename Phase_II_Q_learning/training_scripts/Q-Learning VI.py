
# coding: utf-8

# State: 1-D array of image + coordinates of previous bbox
# Action: Left, Right, None
# 
# In this second phase, we pass a 1-D array of the image, in which one cell indicates the position of the object. Along with this, we include the one-hot vector of the previous bounding box in the state. 
# 
# Thus, we have a very simple network of this form:
# 
# 
#        i1             
#                  h11         h21        left_prob
#        i2       
#                  h12         h22   
#        ..       
#                  h13         h23        none_prob
#        i10       
#        
#        p1        ...         ...
#        
#        p2                               right_prob
#        
#        ..        h110        h210
#        
#        p10
#        
# 
# We have a neural network consisting of only fully connected layers of dimension 20 x 10 x 10 x 3. Here, we have 20 input units (corresponding to the board state and the previous bbox one-hot vector), 10 units in the first hidden layer, 10 units in the second hidden layer, and 3 units 
# 
# The outputs consist of the probabilities of going right, left and staying at the same position. The action then gives the direction in which this previous bounding box should move in order to coincide with the current actual bounding box.
# 
# This method trains on trajectories for which labels for only the first and the last frame are provided. After each episode terminates, the reward is given by the negative of the L-1 distance between the bounding box and the actual object position. For all intermediate steps, which do not have a label associated with them, the reward is set to 0.

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


num_positions = 10
input_size = 20 # [board configuration:10, one-hot previous bbox vector:10]
hidden_layer_1_size = 10
hidden_layer_2_size = 10
output_size = 3 # number of actions = 3 : Left, None, Right
num_actions = 3
gamma = 0.9
alpha = 0.01
epsilon = 0.2
num_epochs = 100


# ### Making the neural network

# In[3]:


X = tf.placeholder('float', [None, input_size])

weights = {
    'h1': tf.Variable(tf.random_normal([input_size, hidden_layer_1_size])),
    'h2': tf.Variable(tf.random_normal([hidden_layer_1_size, hidden_layer_2_size])),
    'out': tf.Variable(tf.random_normal([hidden_layer_2_size, output_size]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([hidden_layer_1_size])),
    'b2': tf.Variable(tf.random_normal([hidden_layer_2_size])),
    'out': tf.Variable(tf.random_normal([output_size]))
}


# In[4]:


def network(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.nn.softmax(tf.nn.relu(tf.matmul(layer_2, weights['out']) + biases['out']))
    return out_layer


# In[5]:


output = network(X)
actual_return = tf.Variable([[0.0, 0.0, 0.0]], name='actual_return', dtype=tf.float32, validate_shape=False)
expected_return = tf.placeholder('float', [None, 3], name='expected_return')
loss = tf.reduce_mean(tf.square(tf.subtract(actual_return, expected_return)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train_op = optimizer.minimize(loss)

saver = tf.train.Saver()


# ### Generate 100000 trajectories

# In[6]:


training_trajectories = []

for i in xrange(10000):
    obj_start = np.random.randint(0, num_positions)
    obj_start = [0 if x != obj_start else 1 for x in xrange(num_positions)]
    obj = obj_start
    trajectory = [obj]
    for j in xrange(10):
        a = np.random.randint(0, num_actions)
        obj_coord = obj.index(1)
        while (obj_coord == 0 and a == 0) or (obj_coord == num_positions - 1 and a == 2):
            a = np.random.randint(0, num_actions)
        obj_coord = obj_coord - 1 if a == 0 else obj_coord if a == 1 else obj_coord + 1
        obj = [0 if x != obj_coord else 1 for x in xrange(num_positions)]
        trajectory.append(obj)
    training_trajectories.append(trajectory)


# ### Training the network

# In[7]:


def epsilon_greedy(actions):
    if np.random.rand() <= epsilon:
        return np.argmax(actions)
    return np.random.randint(0, len(actions))


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for k in xrange(num_epochs):
        print 'Epoch ' + str(k)
	print
        
	for i in xrange(len(training_trajectories)):
            j = 1
            prev_box = training_trajectories[i][0]
            
            states = []
            actions = []
            actual_returns = []
            expected_returns = []
            
            # Evaluate the Q-network to get the Q-values, and on the basis of that, select an action, and
            # consequently calculate the current box coordinates from the action on the previous box coordinates
            while j < len(training_trajectories[i]):
                current_state = training_trajectories[i][j] + prev_box
                current_qvalues = sess.run([output], feed_dict={X:[current_state]})[0][0]
                current_action = epsilon_greedy(current_qvalues)
                prev_box_coord = prev_box.index(1)
                current_box_coord = prev_box_coord - 1 if current_action == 0 else prev_box_coord if current_action == 1 else prev_box_coord + 1
                current_box = [0 if x != current_box_coord else 1 for x in xrange(num_positions)]
                
                # If the current box coordinates are invalid (out of bounds), we set a penalty for them and backpropagate.
                # Otherwise, we add the new state and action to their respective arrays.
                if current_box_coord < 0 or current_box_coord >= num_positions:
                    sess.run(tf.assign(actual_return, [0.0 if x == current_action else 0.0 for x in xrange(num_actions)], validate_shape=False))
                    sess.run([train_op], feed_dict={expected_return:[[0.0 for x in xrange(num_actions)]]})
                    j = 1
                    states = []
                    actions = []
                    actual_returns = []
                    prev_box = training_trajectories[i][0]
                else:
                    states.append(current_state)
                    actions.append(current_action)
                    actual_returns.append([current_qvalues[x] if x == current_action else 0.0 for x in xrange(num_actions)])
                    prev_box = current_box
                    j += 1
                    
            # Assign rewards. Here, the reward is 0 if the end point is the same as the starting point and
            # -distance between the final actual and predicted bounding boxes. The reward is 0 for all intermediate steps.
            rewards = np.zeros(len(training_trajectories[i]) - 1, dtype=np.float32)
            rewards[-1] = -abs(1.0 * current_box_coord - training_trajectories[i][-1].index(1))
            prev_box = training_trajectories[i][0]
            
            for j in xrange(len(training_trajectories[i]) - 2, -1, -1):
                current_state = states[j]
                current_action = actions[j]
                prev_box = current_state[num_positions:]
                prev_box_coord = prev_box.index(1)
                current_box_coord = prev_box_coord - 1 if current_action == 0 else prev_box_coord if current_action == 1 else prev_box_coord + 1
                current_box = [0 if x != current_box_coord else 1 for x in xrange(num_positions)]

                current_qvalues = actual_returns[j]

                next_state = current_state[:num_positions] + current_box
                next_qvalues = sess.run(output, feed_dict={X:[next_state]})[0]
                max_action = np.argmax(next_qvalues)
                
                # We will only update weights for the max action chosen for the next state, so all other actions are made to
                # have the same output as the previous output for Q-values so that their loss is 0 and thus not updated.
                target_qvalues = [rewards[j] + gamma * next_qvalues[x] if x == max_action else current_qvalues[x] for x in xrange(len(next_qvalues))]

                expected_returns.append(target_qvalues)

		print target_qvalues - current_qvalues
                        
            # Train!
            sess.run(tf.assign(actual_return, actual_returns, validate_shape=False))
            sess.run([train_op], feed_dict={expected_return:expected_returns})
            
            if (i + 1) % 2000 == 0:
                print str(i + 1)
                saver.save(sess, '../QL_V_ckpts/relu/' + str(k), global_step=k*len(training_trajectories) + i + 1)
            
            epsilon = 1/(5+0.01*(k * len(training_trajectories) + i + 1))

	    print (i + 1)
            

