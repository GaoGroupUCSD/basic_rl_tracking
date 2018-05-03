import numpy as np
import cv2
import tensorflow as tf

num_positions = 10
input_size = 20 # [board configuration:10, one-hot previous bbox vector:10]
hidden_layer_1_size = 10
hidden_layer_2_size = 10
output_size = 3 # number of actions = 3 : Left, None, Right
num_actions = 3
gamma = 0.9
alpha = 0.01
epsilon = 0.2

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

def network(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.nn.softmax(tf.matmul(layer_2, weights['out']) + biases['out'])
    return out_layer

output = network(X)
actual_return = tf.Variable([[0.0, 0.0, 0.0]], name='actual_return', dtype=tf.float32, validate_shape=False)
expected_return = tf.placeholder('float', [None, 3], name='expected_return')
loss = tf.reduce_mean(tf.square(tf.subtract(actual_return, expected_return)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train_op = optimizer.minimize(loss)

saver = tf.train.Saver()

test_trajectories = []

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
    test_trajectories.append(trajectory)


total = 10000
correct = 0.0

with tf.Session() as sess:
    saver.restore(sess, '../QL_V_ckpts/softmax/qlV.ckpt-16000')
    for i in xrange(len(test_trajectories)):
        j = 1
        prev_box = test_trajectories[i][0]
        img = np.zeros((100, 100, 3))
        cv2.rectangle(img, (test_trajectories[i][0].index(1) * 10, 50), (test_trajectories[i][0].index(1) * 10 + 10, 60), (255, 255, 255), -1)

        cv2.rectangle(img, (test_trajectories[i][0].index(1) * 10, 50), (test_trajectories[i][0].index(1) * 10 + 10, 60), (0, 0, 255), 2)
        cv2.imshow('ql5', img)
	cv2.waitKey(100)


        # Evaluate the Q-network to get the Q-values, and on the basis of that, select an action, and
        # consequently calculate the current box coordinates from the action on the previous box coordinates
        while j < len(test_trajectories[i]):
            current_state = test_trajectories[i][j] + prev_box
            current_qvalues = sess.run([output], feed_dict={X:[current_state]})[0][0]
            current_action = np.argmax(current_qvalues)
            prev_box_coord = prev_box.index(1)
            current_box_coord = prev_box_coord - 1 if current_action == 0 else prev_box_coord if current_action == 1 else prev_box_coord + 1
            current_box = [0 if x != current_box_coord else 1 for x in xrange(num_positions)]

	    img = np.zeros((100, 100, 3))
            cv2.rectangle(img, (test_trajectories[i][j].index(1) * 10, 50), (test_trajectories[i][j].index(1) * 10 + 10, 60), (255, 255, 255), -1)
            print test_trajectories[i][j].index(1)

	    cv2.rectangle(img, (current_box_coord * 10, 50), (current_box_coord * 10 + 10, 60), (0, 0, 255), 2)
	    cv2.imshow('ql5', img)
	    cv2.waitKey(100)

            # If the current box coordinates are invalid (out of bounds), we set a penalty for them and backpropagate.
            # Otherwise, we add the new state and action to their respective arrays.
            if current_box_coord < 0 or current_box_coord >= num_positions:
                break
            else:
                prev_box = current_box
                j += 1
        
        if current_box_coord == test_trajectories[i][-1].index(1):
            correct += 1

        print
            
print correct * 1.0 / total

cv2.destroyAllWindows()
