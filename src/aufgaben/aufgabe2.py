import numpy as np

from src.bandit_factory.bandit_factory import BanditFactory
from src.rewarder.reinforcementagent import ReinforcementAgent

if __name__ == "__main__":
    N_steps = 10000
    k_arms = 10

    bandit_function = BanditFactory.get_normal_bandit_function(k_arms)

    agent = ReinforcementAgent()
    agent.set_bandit(bandit_function)

    for step in range(N_steps):
        random_arm = np.random.randint(k_arms)
        agent.updateQN(random_arm)

    print(bandit_function.get_middle_of_arms())
    print(agent.get_expectations())
