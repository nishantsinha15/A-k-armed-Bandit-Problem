# import statements
import numpy as np
import matplotlib.pyplot as plt

steps = 1000
variance = 1
k = 10
iters = 1


def estimate_q(q_old, reward, n):
    # incremental implementation
    q_new = (q_old + (1.0 / n) * (reward - q_old))
    return q_new


def get_reward(q_star, action):
    return np.random.normal(q_star[action], variance)


# Solves one k-arm bandit problem using greedy approach
class Greedy:

    def __init__(self, q_star):
        self.q_star = q_star
        self.q_estimate = np.zeros(k)
        self.action_count = np.zeros(k)

    def solve(self):
        # greedily select the arm with biggest reward
        for i in range(steps):
            greedy_action = 0
            greedy_reward = -1
            for j in range(k):
                if self.q_estimate[j] > greedy_reward:
                    greedy_action = j
                    greedy_reward = self.q_estimate[j]

            # Get reward from the greedily selected arm
            actual_reward = get_reward(self.q_star, greedy_action)
            self.action_count[greedy_action] += 1
            self.q_estimate[greedy_action] = \
                estimate_q(self.q_estimate[greedy_action], actual_reward, self.action_count[greedy_action])

    def get_solution(self):
        print(self.action_count)
        return self.q_estimate


for run in range(iters):
    q_star = np.random.normal(0, variance, 10)
    greedy_solution = Greedy(q_star)
    greedy_solution.solve()
    print(greedy_solution.get_solution())