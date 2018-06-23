class MultiPolyFunc(object):

    def __init__(self, poly_func, param_count):
        """
        Defining Multi-variate Polynomial Function for Simulated Annealing

        :param poly_func: <Function> Function definition for polynomial
        :param param_count: <Integer> Number of parameters
        """
        self.func = poly_func
        self.param_count = param_count
