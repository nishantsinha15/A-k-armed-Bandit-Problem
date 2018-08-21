# import statements
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

steps = 1000
variance = 1
mean = 4
k = 10
iters = 200


def get_reward(q_star, action):
    return np.random.normal(q_star[action], variance)


# Solves one k-arm bandit problem using greedy approach
class Gradient_Bandit:

    def __init__(self, q_star, alpha):
        self.q_star = q_star
        self.action_count = np.zeros(k)
        self.reward = np.zeros(steps)
        self.correct_action = np.zeros(steps)
        self.prob = np.zeros(k)
        self.alpha = alpha

    def calculate_probability(self, h):
        denom = 0.0
        for i in range(k):
            denom += math.exp(h[i])

        for i in range(k):
            self.prob[i] = math.exp(h[i]) / denom

    def update_using_baseline(self, h, greedy_action, actual_reward, sum_rewards, i):
        for i in range(k):
            if i != greedy_action:
                h[i] = h[i] - self.alpha * ( actual_reward - (sum_rewards/(i+1))) * self.prob[i]
            else:
                h[i] = h[i] + self.alpha * ( actual_reward - (sum_rewards/(i+1))) * (1 - self.prob[i])
        return h

    def solve(self):
        h = np.zeros(k)
        sum_rewards = 0.0
        for i in range(steps):
            greedy_action = np.argmax(self.prob)

            # Get reward from the greedily selected arm
            actual_reward = get_reward(self.q_star, greedy_action)
            sum_rewards += actual_reward
            self.action_count[greedy_action] += 1

            # updation of h
            h = self.update_using_baseline(h, greedy_action, actual_reward, sum_rewards, i )
            self.calculate_probability(h)

            # updating reward at this step
            self.reward[i] = actual_reward

            # checking if it was the optimal reward
            if greedy_action == np.argmax(self.q_star):
                self.correct_action[i] = 1

    def get_correct_action_count(self):
        return self.correct_action


def steps_vs_optimal_action():
    optimal_action_gradient = np.zeros(steps)

    for run in range(iters):
        q_star = np.random.normal(mean, variance, k)

        solution = Gradient_Bandit(q_star, 0.1)
        solution.solve()

        optimal_action_gradient = np.add(optimal_action_gradient, solution.get_correct_action_count())

    optimal_action_gradient = np.divide(optimal_action_gradient, iters)

    plt.plot(optimal_action_gradient, 'r' )
    plt.show()

steps_vs_optimal_action()