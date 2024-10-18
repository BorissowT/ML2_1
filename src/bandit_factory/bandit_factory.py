from src.bandit.bandit import NormalBandit, IBandit


class BanditFactory:

    @staticmethod
    def get_normal_bandit_function(number_of_arms: int):
        """

        :param number_of_arms: int - number of arms for the bandit.
        :return: bandit function: IBandit function that can be played.
        """
        bandit: IBandit = NormalBandit()
        bandit.init_number_of_arms(number_of_arms)
        return bandit
