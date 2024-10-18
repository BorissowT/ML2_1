from abc import ABC, abstractmethod

import numpy as np


class IBandit(ABC):
    @abstractmethod
    def init_number_of_arms(self, number_of_arms:int):
        pass

    @abstractmethod
    def play_arm(self, arms_number:int) -> float:
        pass

    @abstractmethod
    def get_arms(self)-> list:
        pass

class NormalBandit(IBandit):
    """Implements Bandit function. Implements Normal distribution
    with 0 as middle and 1 var."""

    number_of_arms = 0
    arms = []

    def get_arms(self) -> list:
        return self.arms

    def init_number_of_arms(self, number_of_arms: int):
        self.number_of_arms = number_of_arms
        self.arms = np.random.normal(0, 1, number_of_arms)

    def play_arm(self, arms_number: int) -> float:
        return np.random.normal(self.arms[arms_number], 1)