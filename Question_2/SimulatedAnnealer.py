import numpy as np
import random


class SimulatedAnnealer (object):
    def __init__(self, t_range, trials, anneal_obj, prob_obj, minimize, anneal_obj_range):
        """
        :param t_range: <Double[]> Contains the range of temperatures
        :param trials: <Integer> The number of trials to run for every temperature specified in t_range
        :param anneal_obj: <Object> Cost Function to apply Simulated Annealing
        :param prob_obj: <Object> Cost Probability Distribution Function
        :param minimize: <Boolean> True means minimize, False means maximize
        :param anneal_obj_range: <Integer[]> Range of annealing object input parameters [LB1, UB1, LB2, UB2, ...]
        """
        self.t_range = t_range
        self.trials = trials
        self.anneal_obj = anneal_obj
        self.prob_obj = prob_obj
        self.minimize = minimize
        self.anneal_obj_range = anneal_obj_range

        print("Project Initialized")

    def anneal(self):
        print("Starting Annealing Process...")

        for temperature in self.t_range:
            for trial in range(self.trials):

                print("Trial per temperature")

