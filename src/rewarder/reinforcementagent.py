from src.bandit.bandit import IBandit


class ReinforcementAgent:

    bandit: IBandit = None
    N_arms = []
    Q_arms = []

    def get_expectations(self):
        return [(f"arm_{i}", result) for i ,result in enumerate(self.Q_arms)]

    def set_bandit(self, bandit: IBandit):
        self.bandit = bandit
        self._set_N_for_arms()
        self._set_Q_for_arms()

    def updateQN(self, arm: int):
        reward = self.bandit.play_arm(arm)
        self._update_N(arm)
        self._update_Q(arm, reward)

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


