import numpy as np

LIMITS = (0, 9)
NUM_ACTIONS = 3
MAX_STEPS = 10

class env():
    def __init__(self):
        self.steps = 0
        self.prev = 0
        self.board = 0

    # Checks if the box is within the bounds of the board.
    def check(self, box):
        if box < LIMITS[0] or box > LIMITS[1]:
            return False
        return True

    # Converts the box/board number into its one-hot representation.
    def one_hot(self, b):
        return [0 if x != b else 1 for x in xrange(LIMITS[1] + 1)]
    
    # Initializes the env variables for a new trajectory.
    def first_time_step(self):
        self.steps = 0
        self.prev = np.random.randint(0, LIMITS[1] + 1)
        self.board = self.prev
        return self.next_time_step()

    # Generates the next time step in the trajectory.
    def next_time_step(self):
        self.steps += 1
        action = np.random.randint(0, NUM_ACTIONS)
        new_board = self.get_next_box(self.board, action)
        while not self.check(new_board):
            action = np.random.randint(0, NUM_ACTIONS)
            new_board = self.get_next_box(self.board, action)
        self.board = new_board
        state = self.one_hot(self.board) + self.one_hot(self.prev)
        return state
    
    # Returns the current state in the trajectory.
    def get_current_state(self):
        return self.one_hot(self.board) + self.one_hot(self.prev)

    # Performs action specified by the user on the current state.
    def perform_action(self, a):
        new_box = self.get_next_box(self.prev, a)
        self.prev = new_box
        return

    # Checks if the trajectory is terminated. This could happen in 2 cases:
    # 1. The max. number of steps has been reached.
    # 2. The prediction of the user is out of bounds.
    def done(self):
        if self.steps == MAX_STEPS or not self.check(self.prev):
            return True
        return False

    # Performs action a on the current box b, and returns the new box.
    def get_next_box(self, b, a):
        b = b - 1 if a == 0 else b if a == 1 else b + 1
        if self.check(b):
            return b
        return -1

    # Returns the reward at the end of the trajectory.
    # If the trajectory ended because the box went out of bounds, a high penalty is imposed.
    # Otherwise, the penalty imposed just equals the distance between the prediction and the actual position.
    def get_reward(self):
        if self.check(self.prev):
            return -abs(self.prev - self.board)
        return -(LIMITS[1] * 100)

    # Returns the current board position (single integer).
    def get_board(self):
        return self.board

    # Returns the previous box position (single integer).
    def get_box(self):
    	return self.prev
