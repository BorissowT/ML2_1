import time

import numpy as np
import matplotlib.pyplot as plt


from src.bandit_factory.bandit_factory import BanditFactory

if __name__ == "__main__":
    k = 10
    Ne = 500
    N = 1000

    all_rewards = np.zeros(N)

    for episode in range(Ne):
        bandit_function = BanditFactory.get_normal_bandit_function(k)
        episode_rewards = np.zeros(N)

        for step in range(N):
            random_arm = np.random.randint(k)
            episode_rewards[step] = bandit_function.play_arm(random_arm)

        all_rewards += episode_rewards

    average_rewards = all_rewards / Ne

    plt.plot(average_rewards)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title(f'{k}-Armed Bandit: Average Reward over {Ne} Episodes')
    plt.show()
