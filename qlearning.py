import numpy as np

# Q-learning for the pursuer
class QLearning:
    def __init__(self, num_actions, learning_rate, discount_factor, epsilon):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = [0] * self.num_actions
        return self.q_table[state][action]

    def choose_action(self, state):
        if state not in self.q_table:
            action = np.random.choice(range(self.num_actions))
        else:
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(range(self.num_actions))
            else:
                action = np.argmax(self.q_table[state])
        return action

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        max_next_q = max(self.get_q_value(next_state, a) for a in range(self.num_actions))
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q