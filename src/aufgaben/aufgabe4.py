import numpy as np
import matplotlib.pyplot as plt
from src.bandit_factory.bandit_factory import BanditFactory
from src.agent.reinforcementagent import ReinforcementAgent

if __name__ == "__main__":
    N_steps = 1000
    k_arms = 10
    N_episodes = 500
    eps = 0.5
    q_0 = 5
    alpha_values = [0.1, 0.05, 0.01]

    plt.figure(figsize=(12, 8))

    for alpha in alpha_values:
        rewards_per_step = np.zeros(N_steps)

        for episode in range(N_episodes):
            bandit_function = BanditFactory.get_normal_bandit_function(k_arms)
            agent = ReinforcementAgent(q_0)
            agent.set_bandit(bandit_function)

            for step in range(N_steps):
                if np.random.rand() < eps:
                    arm = np.random.randint(k_arms)
                else:
                    arm = np.argmax(agent.Q_arms)

                agent.updateQN_with_alpha(arm, alpha)

            rewards = np.array(agent.get_step_rewards(), dtype=np.float64)
            rewards_per_step += rewards

        avg_rewards_per_step = rewards_per_step / N_episodes

        plt.plot(avg_rewards_per_step, label=f'α={alpha}, ε={eps}, q₀={q_0}')

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward per Step with Constant Step Size over"
              f" {N_episodes} Episodes")
    plt.legend()
    plt.show()
