import numpy as np


class LogisticInstance:
    #-------------------------------------------------------------------------------#
    #       This class defines the following logistic issue.                        #
    #   We are to maximize the probabilty of selling one particular product         #
    #   and the expected profit of it. The issue is determined by the               #
    #   parameters a, b, c, d so that                                               #
    #     --> the probability of selling directly depends on a^Tx + b               #
    #                                      (we call this value 'merchantability'),  #
    #     --> the profit of selling equals c^Tx + d,                                #
    #   where x we are to optimize over is a vector of variables that effect on it. #
    #   x also has to satisfy linear constraint Fx <= g                             #
    #                                                                               #
    #   Parameters:                                                                 #
    #     a -- 1D np.array or list of shape (n,)                                    #
    #     b -- number                                                               #
    #     c -- 1D np.array or list with the same shape as a                         #
    #     d -- number                                                               #
    #     F -- 2D matrix (np.array or list) of shape (n, m)                         #
    #     g -- 1D np.array or list of shape (m,)                                    #
    #-------------------------------------------------------------------------------#
    
    def __init__(self, a, b, c, d, F, g):
        assert(isinstance(b, (float, int)) and 
               isinstance(d, (float, int)) and 
               isinstance(a, (np.ndarray, list)) and
               isinstance(c, (np.ndarray, list)) and
               isinstance(F, (np.ndarray, list)) and
               isinstance(g, (np.ndarray, list)))

        self.a = np.array(a)
        self.b = b
        self.c = np.array(c)
        self.d = d
        self.F = F
        self.g = g

        assert(self.a.shape == self.c.shape and
               len(self.a.shape) == 1 and
               len(self.g.shape) == 1 and
               len(self.F.shape) == 2 and
               self.g.shape[0] == self.F.shape[0] and
               self.F.shape[1] == self.a.shape[0])


    def merchantability(self, x):
        assert(isinstance(x, np.ndarray) and x.shape == self.a.shape)

        return np.dot(self.a, x) + self.b

    def profit(self, x):
        assert(isinstance(x, np.ndarray) and x.shape == self.c.shape)
        
        return np.dot(self.c, x) + self.d

    def prob(self, x):
        return np.exp(self.merchantability(x)) / (1. + np.exp(self.merchantability(x)))

    def expected_profit(self, x):
        return self.profit(x) * self.prob(x)


    def log_expected_profit(self, x):
        return np.log(np.abs(self.profit(x))) + self.merchantability(x) - \
               np.log(1. + np.exp(self.merchantability(x)))

    def dmerchantability(self, x):
        assert(isinstance(x, np.ndarray) and x.shape == self.a.shape)
        
        return self.a

    def dlog_expected_profit(self, x):
        return self.c / self.profit(x) + self.a * (1. - self.prob(x))
