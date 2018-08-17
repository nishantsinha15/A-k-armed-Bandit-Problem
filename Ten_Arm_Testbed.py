# import statements
import numpy as np
import matplotlib.pyplot as plt

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


# Solves one k-arm bandit problem using greedy approach
class Greedy:

    def __init__(self, q_star):
        self.q_star = q_star
        self.q_estimate = np.zeros(k)
        self.action_count = np.zeros(k)
        self.reward = np.zeros(steps)

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

            #updating reward at this step
            self.reward[i] = actual_reward

    def get_q_estimate(self):
        print(self.action_count)
        return self.q_estimate

    def get_reward(self):
        return self.reward


# Solves one k-arm bandit problem using epsilon-greedy approach
class Epsilon_greedy:

    def __init__(self, q_star, epsilon = 0.1):
        self.q_star = q_star
        self.q_estimate = np.zeros(k)
        self.action_count = np.zeros(k)
        self.reward = np.zeros(steps)
        self.epsilon = epsilon

    def solve(self):
        # Epsilon greedy Method
        for i in range(steps):
            greedy_action = 0
            greedy_reward = -1

            random_variable = np.random.uniform(0,1)
            if( random_variable <= self.epsilon ):
                greedy_action = np.random.uniform(0,1)
                greedy_action *= 10
                greedy_action = int(greedy_action)
            else:
                for j in range(k):
                    if self.q_estimate[j] > greedy_reward:
                        greedy_action = j
                        greedy_reward = self.q_estimate[j]

            # Get reward from the (not so)greedily selected arm
            actual_reward = get_reward(self.q_star, greedy_action)
            self.action_count[greedy_action] += 1
            self.q_estimate[greedy_action] = \
                estimate_q(self.q_estimate[greedy_action], actual_reward, self.action_count[greedy_action])

            #updating reward at this step
            self.reward[i] = actual_reward

    def get_q_estimate(self):
        print(self.action_count)
        return self.q_estimate

    def get_reward(self):
        return self.reward


def steps_vs_average_reward():
    avg_reward_greedy = np.zeros(steps)
    avg_reward_epsilon_1 = np.zeros(steps)
    avg_reward_epsilon_01 = np.zeros(steps)

    for run in range(iters):
        q_star = np.random.normal(0, variance, 10)

        greedy_solution = Greedy(q_star)
        greedy_solution.solve()
        avg_reward_greedy = np.add(avg_reward_greedy, greedy_solution.get_reward())

        epsilon_1 = Epsilon_greedy(q_star, 0.1)
        epsilon_1.solve()
        avg_reward_epsilon_1 = np.add(avg_reward_epsilon_1, epsilon_1.get_reward())

        epsilon_01 = Epsilon_greedy(q_star, 0.01)
        epsilon_01.solve()
        avg_reward_epsilon_01 = np.add(avg_reward_epsilon_01, epsilon_01.get_reward())

    avg_reward_greedy = np.divide(avg_reward_greedy, iters)
    avg_reward_epsilon_1 = np.divide(avg_reward_epsilon_1, iters)
    avg_reward_epsilon_01 = np.divide(avg_reward_epsilon_01, iters)

    plt.plot(avg_reward_greedy, 'r', avg_reward_epsilon_1, 'b', avg_reward_epsilon_01, 'g')
    plt.show()


steps_vs_average_reward()