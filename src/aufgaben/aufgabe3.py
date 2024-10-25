import numpy as np

from src.bandit_factory.bandit_factory import BanditFactory
from src.rewarder.reinforcementagent import ReinforcementAgent

if __name__ == "__main__":
    N_steps = 1000
    k_arms = 10

    bandit_function = BanditFactory.get_normal_bandit_function(k_arms)

    agent = ReinforcementAgent()
    agent.set_bandit(bandit_function)

    agent.updateQN_with_eps_greedy(eps=0.5, N=N_steps)

    print(bandit_function.get_middle_of_arms())
    print(agent.get_expectations())
    print(agent.get_arm_total_calls())
