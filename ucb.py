# import statements
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


steps = 1000
variance = 1
k = 10
iters = 2000


def estimate_q(q_old, reward, n):
    # incremental implementation
    q_new = (q_old + (1.0 / n) * (reward - q_old))
    return q_new

def get_reward(q_star, action):
    return np.random.normal(q_star[action], variance)

# Solves one k-arm bandit problem using epsilon-greedy approach
class Epsilon_greedy:

    def __init__(self, q_star, epsilon=0.1):
        self.q_star = q_star
        self.q_estimate = np.zeros(k)
        self.action_count = np.zeros(k)
        self.reward = np.zeros(steps)
        self.epsilon = epsilon
        self.correct_action = np.zeros(steps)
        self.absolute_error = np.zeros((k, steps))

    def solve(self):
        # Epsilon greedy Method
        for i in range(steps):
            random_variable = np.random.uniform(0, 1)

            # explore
            if (random_variable <= self.epsilon):
                greedy_action = np.random.uniform(0, 1)
                greedy_action *= 10
                greedy_action = int(greedy_action)

            # exploit
            else:
                # greedily select the arm with biggest reward
                greedy_action = np.argmax(self.q_estimate)

            # Get reward from the (not so)greedily selected arm
            actual_reward = get_reward(self.q_star, greedy_action)
            self.action_count[greedy_action] += 1
            self.q_estimate[greedy_action] = \
                estimate_q(self.q_estimate[greedy_action], actual_reward, self.action_count[greedy_action])

            # updating reward at this step
            self.reward[i] = actual_reward

            # checking if it was the optimal reward
            if greedy_action == np.argmax(self.q_star):
                self.correct_action[i] = 1

            # finding the absolute error
            for j in range(k):
                self.absolute_error[j][i] = abs(self.q_star[j] - self.q_estimate[j])

    def get_absolute_error(self):
        return self.absolute_error

    def get_q_estimate(self):
        return self.q_estimate

    def get_reward(self):
        return self.reward

    def get_correct_action_count(self):
        return self.correct_action


# Solves one k-arm bandit problem using greedy approach
class UCB:

    def get_ucb_max(self, step):
        action = -1
        temp = -1
        for i in range(k):
            if self.get_ucb_value(i, step) > temp:
                temp = self.get_ucb_value(i, step)
                action = i
        return action

    def get_ucb_value(self, action, step):
        if self.action_count[action] == 0:
            return sys.maxsize
        else:
            return (self.q_estimate[action] + self.c * math.sqrt(math.log(step) / self.action_count[action]))

    def __init__(self, q_star, c):
        self.q_star = q_star
        self.q_estimate = np.zeros(k)
        self.action_count = np.zeros(k)
        self.reward = np.zeros(steps)
        self.correct_action = np.zeros(steps)
        self.absolute_error = np.zeros((k, steps))
        self.c = c

    def solve(self):
        for i in range(steps):

            # greedily select the arm with biggest reward
            greedy_action = self.get_ucb_max(i)

            # Get reward from the greedily selected arm
            actual_reward = get_reward(self.q_star, greedy_action)
            self.action_count[greedy_action] += 1
            self.q_estimate[greedy_action] = \
                estimate_q(self.q_estimate[greedy_action], actual_reward, self.action_count[greedy_action])

            # updating reward at this step
            self.reward[i] = actual_reward

            # checking if it was the optimal reward
            if greedy_action == np.argmax(self.q_star):
                self.correct_action[i] = 1

            # finding the absolute error
            for j in range(k):
                self.absolute_error[j][i] = abs(self.q_star[j] - self.q_estimate[j])

    def get_absolute_error(self):
        return self.absolute_error

    def get_q_estimate(self):
        print(self.action_count)
        return self.q_estimate

    def get_reward(self):
        return self.reward

    def get_correct_action_count(self):
        return self.correct_action

def create_objects():
    q_star = np.random.normal(0, variance, 10)

    ucb_1 = UCB(q_star, c = 1)
    ucb_1.solve()
    ucb_2 = UCB(q_star, c = 2)
    ucb_2.solve()
    ucb_4 = UCB(q_star, c = 4)
    ucb_4.solve()

    epsilon_1 = Epsilon_greedy(q_star, 0.1)
    epsilon_1.solve()

    return ucb_1, ucb_2, ucb_4, epsilon_1

def steps_vs_average_reward():
    avg_reward_ucb_2 = np.zeros(steps)
    avg_reward_ucb_1 = np.zeros(steps)
    avg_reward_ucb_4 = np.zeros(steps)
    avg_reward_epsilon_1 = np.zeros(steps)

    for run in range(iters):
        ucb_1, ucb_2, ucb_4, epsilon_1 = create_objects()

        avg_reward_ucb_1 = np.add(avg_reward_ucb_1, ucb_1.get_reward())
        avg_reward_ucb_2 = np.add(avg_reward_ucb_2, ucb_2.get_reward())
        avg_reward_ucb_4 = np.add(avg_reward_ucb_4, ucb_4.get_reward())

        avg_reward_epsilon_1 = np.add(avg_reward_epsilon_1, epsilon_1.get_reward())

    avg_reward_ucb_1 = np.divide(avg_reward_ucb_1, iters)
    avg_reward_ucb_2 = np.divide(avg_reward_ucb_2, iters)
    avg_reward_ucb_4 = np.divide(avg_reward_ucb_4, iters)
    avg_reward_epsilon_1 = np.divide(avg_reward_epsilon_1, iters)

    print(avg_reward_ucb_2)
    plt.plot(avg_reward_ucb_1, 'r', avg_reward_epsilon_1, 'b', avg_reward_ucb_2, 'g', avg_reward_ucb_4, 'y')
    plt.show()

steps_vs_average_reward()