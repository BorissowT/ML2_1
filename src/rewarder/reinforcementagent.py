import random

import numpy as np

from src.bandit.bandit import IBandit


class ReinforcementAgent:

    bandit: IBandit = None
    N_arms = []
    Q_arms = []
    q0 = 0

    def __init(self, q_0=0):
        self.q0 = q_0

    def get_expectations(self):
        return [(f"arm_{i}", result) for i ,result in enumerate(self.Q_arms)]

    def get_arm_total_calls(self):
        return [(f"arm_{i}", result) for i ,result in enumerate(self.N_arms)]

    def set_bandit(self, bandit: IBandit):
        self.bandit = bandit
        self._set_N_for_arms()
        self._set_Q_for_arms()

    def updateQN(self, arm: int):
        reward = self.bandit.play_arm(arm)
        self._update_N(arm)
        self._update_Q(arm, reward)

    def updateQN_with_eps_greedy(self, eps:float, N: int):
        self._validate_eps(eps)
        for _ in range(N):
            z = random.uniform(0, 1)
            if z < eps:
                random_arm = np.random.randint(
                    self.bandit.get_number_of_arms()
                )
                self.updateQN(random_arm)
            elif z >= eps:
                max_value = max(self.Q_arms)
                max_arm = self.Q_arms.index(max_value)
                self._update_N(max_arm)


    def _update_N(self, arm:int):
        self.N_arms[arm] += 1

    def _set_N_for_arms(self):
        number_of_arms = self.bandit.get_number_of_arms()
        for _ in range(number_of_arms):
            self.N_arms.append(0)

    def _set_Q_for_arms(self):
        number_of_arms = self.bandit.get_number_of_arms()
        for _ in range(number_of_arms):
            self.Q_arms.append(0)

    def _update_Q(self, arm, reward):
        self.Q_arms[arm] = (self.Q_arms[arm] +
                          (1 / self.N_arms[arm]) *
                          (reward - self.Q_arms[arm]))

    def _validate_eps(self, eps):
        if eps > 1 or eps < 0:
            raise ValueError("Eps must be in range [0,1]")


