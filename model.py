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
        self.a = np.array(a)
        self.b = b
        self.c = np.array(c)
        self.d = d
        self.F = F
        self.g = g


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

    def d2merchantability(self, x):
        assert(isinstance(x, np.ndarray) and x.shape == self.a.shape)
        
        return np.zeros(x.shape[0])

    def dlog_expected_profit(self, x):
        return self.c / self.profit(x) + self.a * (1. - self.prob(x))

    def d2log_expected_profit(self, x):
        c = np.array([self.c])
        a = np.array([self.a])
        return -(c.T @ c) / (self.profit(x) ** 2) - (a.T @ a) * self.prob(x) / (1. + np.exp(self.merchantability(x)))
