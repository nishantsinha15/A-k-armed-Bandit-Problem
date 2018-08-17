# import statements
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


# Solves one k-arm bandit problem using greedy approach
class Greedy:

    def __init__(self, q_star):
        self.q_star = q_star
        self.q_estimate = np.zeros(k)
        self.action_count = np.zeros(k)
        self.reward = np.zeros(steps)
        self.correct_action = np.zeros(steps)
        self.absolute_error = np.zeros((k, steps))

    def solve(self):
        for i in range(steps):

            # greedily select the arm with biggest reward
            greedy_action = np.argmax(self.q_estimate)

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


def create_objects():
    q_star = np.random.normal(0, variance, 10)

    greedy_solution = Greedy(q_star)
    greedy_solution.solve()

    epsilon_1 = Epsilon_greedy(q_star, 0.1)
    epsilon_1.solve()

    epsilon_01 = Epsilon_greedy(q_star, 0.01)
    epsilon_01.solve()

    return greedy_solution, epsilon_1, epsilon_01


def steps_vs_average_reward():
    avg_reward_greedy = np.zeros(steps)
    avg_reward_epsilon_1 = np.zeros(steps)
    avg_reward_epsilon_01 = np.zeros(steps)

    for run in range(iters):
        greedy_solution, epsilon_1, epsilon_01 = create_objects()

        avg_reward_greedy = np.add(avg_reward_greedy, greedy_solution.get_reward())
        avg_reward_epsilon_1 = np.add(avg_reward_epsilon_1, epsilon_1.get_reward())
        avg_reward_epsilon_01 = np.add(avg_reward_epsilon_01, epsilon_01.get_reward())

    avg_reward_greedy = np.divide(avg_reward_greedy, iters)
    avg_reward_epsilon_1 = np.divide(avg_reward_epsilon_1, iters)
    avg_reward_epsilon_01 = np.divide(avg_reward_epsilon_01, iters)

    plt.plot(avg_reward_greedy, 'r', avg_reward_epsilon_1, 'b', avg_reward_epsilon_01, 'g')
    plt.show()


def steps_vs_optimal_action():
    optimal_action_greedy = np.zeros(steps)
    optimal_action_epsilon_1 = np.zeros(steps)
    optimal_action_epsilon_01 = np.zeros(steps)

    for run in range(iters):
        greedy_solution, epsilon_1, epsilon_01 = create_objects()

        optimal_action_greedy = np.add(optimal_action_greedy, greedy_solution.get_correct_action_count())
        optimal_action_epsilon_1 = np.add(optimal_action_epsilon_1, epsilon_1.get_correct_action_count())
        optimal_action_epsilon_01 = np.add(optimal_action_epsilon_01, epsilon_01.get_correct_action_count())

    optimal_action_greedy = np.divide(optimal_action_greedy, iters)
    optimal_action_epsilon_1 = np.divide(optimal_action_epsilon_1, iters)
    optimal_action_epsilon_01 = np.divide(optimal_action_epsilon_01, iters)

    plt.plot(optimal_action_greedy, 'r', optimal_action_epsilon_1, 'b', optimal_action_epsilon_01, 'g')
    plt.show()


def get_colors(count):
    x = np.arange(count)
    ys = [i + x + (i * x) ** 2 for i in range(count)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    return colors


def step_vs_absolute_error_greedy():
    colors = get_colors(k)
    average_absolute_error = np.zeros((k, steps))
    for run in range(iters):
        greedy_solution, epsilon_1, epsilon_01 = create_objects()

        average_absolute_error = np.add(average_absolute_error, greedy_solution.get_absolute_error())
    average_absolute_error = np.divide(average_absolute_error, iters)
    axes = plt.gca()
    # axes.set_ylim([0, 0.5])
    for val, c in zip(average_absolute_error, colors):
        plt.plot(val, color=c)
    plt.show()

def step_vs_absolute_error_epsilon_1():
    colors = get_colors(k)
    average_absolute_error = np.zeros((k, steps))
    for run in range(iters):
        greedy_solution, epsilon_1, epsilon_01 = create_objects()

        average_absolute_error = np.add(average_absolute_error, epsilon_1.get_absolute_error())
    average_absolute_error = np.divide(average_absolute_error, iters)
    axes = plt.gca()
    # axes.set_ylim([0, 0.5])
    for val, c in zip(average_absolute_error, colors):
        plt.plot(val, color=c)
    plt.show()

def step_vs_absolute_error_epsilon_01():
    colors = get_colors(k)
    average_absolute_error = np.zeros((k, steps))
    for run in range(iters):
        greedy_solution, epsilon_1, epsilon_01 = create_objects()

        average_absolute_error = np.add(average_absolute_error, epsilon_01.get_absolute_error())
    average_absolute_error = np.divide(average_absolute_error, iters)
    axes = plt.gca()
    # axes.set_ylim([0, 0.5])
    for val, c in zip(average_absolute_error, colors):
        plt.plot(val, color=c)
    plt.show()

step_vs_absolute_error_epsilon_1()
step_vs_absolute_error_epsilon_01()