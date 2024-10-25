import numpy as np
import matplotlib.pyplot as plt
from src.bandit_factory.bandit_factory import BanditFactory
from src.rewarder.reinforcementagent import ReinforcementAgent

if __name__ == "__main__":
    N_steps = 1000
    k_arms = 10
    N_episodes = 500  # Minimum number of episodes for averaging
    rewards_per_step = np.zeros(N_steps)

    for episode in range(N_episodes):
        # Initialize bandit and agent for each episode
        bandit_function = BanditFactory.get_normal_bandit_function(k_arms)
        agent = ReinforcementAgent()
        agent.set_bandit(bandit_function)

        # Update with a greedy strategy (eps=0)
        agent.updateQN_with_eps_greedy(eps=0, N=N_steps)

        # Accumulate rewards for each step to calculate the average
        rewards = np.array(agent.get_step_rewards(), dtype=np.float64)
        rewards_per_step += rewards

    # Calculate the average reward for each step across episodes
    avg_rewards_per_step = rewards_per_step / N_episodes

    # Plot the average reward at each step
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_per_step, label='Average Reward (ε=0, Greedy)')
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Step (ε=0, Greedy Strategy) over 500 Episodes")
    plt.legend()
    plt.show()
