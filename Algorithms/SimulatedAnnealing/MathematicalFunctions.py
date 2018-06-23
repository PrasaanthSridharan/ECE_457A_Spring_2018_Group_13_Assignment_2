import math


class MathematicalFunctions:

    __FUNC_PARAM_MAP = {
        "easom_func": 2,
        "s_a_prob_func": 2
    }

    @staticmethod
    def easom_func(x):
        """
        Easom Function Definition

        :param x: <Double[]> Array containing param of length 2.
        :return: <Double> Solution for Easom Function
        """

        return -math.cos(x[0]) * math.cos(x[1]) * math.exp(-(x[0] - math.pi)**2 - (x[1] - math.pi)**2)

    @staticmethod
    def s_a_prob_func(x):
        """
        Default probability function for simulated annealing

        :param x: <Double[]> [0] Change in Cost; [1] Current Temperature
        :return: Returns probability given change in cost and current temperature
        """

        return math.exp(-x[0]/x[1])

    @staticmethod
    def get_function_params(func_name):
        """
        Get the specified function's parameter count

        :param self: <Object> Instance object
        :param func_name: <String> Name of function to get parameter count for
        :return: <Integer> Returns a number containing count of parameter for specified function
        """

        return MathematicalFunctions.__FUNC_PARAM_MAP[func_name]
