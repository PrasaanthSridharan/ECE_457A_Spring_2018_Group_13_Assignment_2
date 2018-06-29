import numpy as np
import random


class SimulatedAnnealer (object):

    def __init__(self, t_range, trials, anneal_obj, anneal_obj_range, prob_obj, is_minimize):
        """
        :param t_range: <Double[]> Contains the range of temperatures
        :param trials: <Integer> The number of trials to run for every temperature specified in t_range
        :param anneal_obj: <Object> Cost Function to apply Simulated Annealing
        :param anneal_obj_range: <Integer[]> Range of annealing object input parameters [LB1, UB1, LB2, UB2, ...]
        :param prob_obj: <Object> Cost Probability Distribution Function. [0] is cost, [1] is temperature
        :param is_minimize: <Boolean> True means minimize, False means maximize
        """
        self.t_range = t_range
        self.trials = trials
        self.anneal_obj = anneal_obj
        self.prob_obj = prob_obj
        self.is_minimize = is_minimize
        self.anneal_obj_range = anneal_obj_range

    def __init__(self, t_start, t_end, t_decrement, is_geometric, trials, anneal_obj, anneal_obj_range, prob_obj,
                 is_minimize):
        """
        :param t_start: <Double> Starting temperature
        :param t_end: <Double> Ending temperature
        :param t_decrement: <Double> Temperature Decrement Value
        :param is_geometric: <Boolean> True means geometric annealing, False means linear annealing
        :param trials: <Integer> The number of trials to run for every temperature specified in t_range
        :param anneal_obj: <Object> Cost Function to apply Simulated Annealing
        :param anneal_obj_range: <Integer[]> Range of annealing object input parameters [LB1, UB1, LB2, UB2, ...]
        :param prob_obj: <Object> Cost Probability Distribution Function. [0] is cost, [1] is temperature
        :param is_minimize: <Boolean> True means minimize, False means maximize
        """
        self.t_start = t_start
        self.t_end = t_end
        self.t_decrement = t_decrement
        self.is_geometric = is_geometric
        self.trials = trials
        self.anneal_obj = anneal_obj
        self.prob_obj = prob_obj
        self.is_minimize = is_minimize
        self.anneal_obj_range = anneal_obj_range

        self.t_range = SimulatedAnnealer.__generate_temp_range(t_start, t_end, t_decrement, is_geometric)

        print("Project Initialized")

    def anneal(self):

        # Initialize annealing object iteration parameters
        x_c = np.zeros(self.anneal_obj.param_count)  # current
        x_b = np.copy(x_c)                           # best
        x_a = np.zeros(self.anneal_obj.param_count)  # aspired

        for temp in self.t_range:
            for trial in range(self.trials):
                x_a = self.__generate_aspiration(x_c, x_a)

                if self.__calc_prob_accept(x_c, x_a, temp):
                    x_c = x_a

                if self.anneal_obj.func(x_c) < self.anneal_obj.func(x_b):
                    x_b = np.copy(x_c)

        print("Best Objective:", self.anneal_obj.func(x_b))
        print("Best Solution:", x_b)

    # Helper Methods ###################################################################################################
    @staticmethod
    def __generate_temp_range(t_start, t_end, t_decrement, is_geometric):
        """
        Generates an array containing the range of temperatures

        :param t_start: <Double> Starting temperature
        :param t_end: <Double> Final temperature
        :param t_decrement: <Double> Decrement value
        :param is_geometric: <Boolean> "True" implements geometric SA, "False" implements linear SA
        :return: <Double[]> Array containing range of temperatures
        """

        t_range = []
        t = t_start

        if is_geometric:
            while t >= t_end:
                t_range.append(t)
                t = t * t_decrement
        else:
            while t >= t_end:
                t_range.append(t)
                t = t - t_decrement

        return t_range

    def __generate_aspiration(self, x_c, x_a, multiplier=1):
        """
        Generate the aspired value based on the current parameters. Also ensures that the functions boundary conditions
        for the parameters are respected when aspiring.

        :param x_c: <Double[]> Current parameters
        :param x_a: <Double[]> Aspiration parameters
        :param multiplier: <Double> Multiplier to aspire more or less quickly
        :return: x_a <Double[]>
        """

        for idx in range(self.anneal_obj.param_count):
            x_a[idx] = x_c[idx] + (random.random() - 0.5) * multiplier
            x_a[idx] = min(max(x_a[idx], self.anneal_obj_range[2*idx]), self.anneal_obj_range[2*idx+1])

        return x_a

    def __calc_prob_accept(self, x_c, x_a, temp):
        """
        If the aspiration objective is more than the current objective, calculates the probability of acceptance a
        Boolean value for whether or not the probability condition was satisfied. Else, "True" is returned.

        :param x_c: <Double[]> Current parameters
        :param x_a: <Double[]> Aspiration parameters
        :param temp: <Double> Current temperature
        :return: <Boolean> "True" if aspire, "False" if not aspire
        """

        if self.anneal_obj.func(x_a) > self.anneal_obj.func(x_c):
            return random.random() < self.prob_obj.func([self.anneal_obj.func(x_c) - self.anneal_obj.func(x_a), temp])
        else:
            return True
    ####################################################################################################################
