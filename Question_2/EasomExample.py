from SimulatedAnnealing.MultiPolyFunc import MultiPolyFunc as mpf
from SimulatedAnnealing.MathematicalFunctions import MathematicalFunctions as mf
from SimulatedAnnealing.SimulatedAnnealer import SimulatedAnnealer as sa



# Test
import random


def main():

    # Define Simulated Annealing Input Parameters
    t_start = 15000
    t_end = 20
    t_decrement = 0.98
    is_geometric = True
    trials = 200
    easom_func = mpf(mf.easom_func, mf.get_function_params("easom_func"))
    easom_func_param_range = [-100, 100, -100, 100]
    prob_func = mpf(mf.s_a_prob_func, mf.get_function_params("s_a_prob_func"))
    is_minimize = True

    # Define Annealer with parameters
    simulated_annealer = sa(t_start, t_end, t_decrement, is_geometric, trials, easom_func, easom_func_param_range,
                            prob_func, is_minimize)

    # Perform annealing process and get converged solution
    simulated_annealer.anneal()

def test():
    print(random.random())


if __name__ == "__main__":
    main()
