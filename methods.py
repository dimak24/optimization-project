import numpy.linalg as la
from enum import Enum
import numpy as np



class Convergence(Enum):
    ByArgument          = lambda x, f, df, d2f, EPS: la.norm(x[-1] - x[-2]) < EPS
    ByValue             = lambda x, f, df, d2f, EPS: la.norm(f(x[-1]) - f(x[-2])) < EPS
    Gradient            = lambda x, f, df, d2f, EPS: la.norm(df(x[-1])) < EPS
    ByArgumentLastThree = lambda x, f, df, d2f, EPS: np.min([la.norm(x[-1] - x[-2]), 
                                                             la.norm(x[-1] - x[-3]), 
                                                             la.norm(x[-2] - x[-3])]) < EPS



class MethodOfOptimization:
    def __init__(self, precision=1e-5, convergence_condition=Convergence.ByArgument):
        self.EPS = precision
        self.convergence_condition__ = convergence_condition
        self.x = []


    def convergence_condition(self, f, df, d2f):
        return self.convergence_condition__(self.x, f, df, d2f, self.EPS)

    def pre_call(self, f, x0, df, d2f, g, dg):
        del self.x[:]
        self.x.append(x0)

    def next_x(self, f, df, d2f, g, dg):
        return self.x[-1]

    def update(self, f, df, d2f, g, dg):
        pass


    def __call__(self, f, x0, df=None, d2f=None, g=None, dg=None):
        self.pre_call(f, x0, df, d2f, g, dg)

        while len(self.x) < 4 or not self.convergence_condition(f, df, d2f):
            self.x.append(self.next_x(f, df, d2f, g, dg))
            self.update(f, df, d2f, g, dg)

        return self.x[-1]



class GradientDescent(MethodOfOptimization):
    def __init__(self, precision=1e-5, convergence_condition=Convergence.ByArgumentLastThree, h=0.5):
        super().__init__(precision, convergence_condition)
        self.h = h if callable(h) else lambda step: h


    def next_x(self, f, df, d2f, g, dg):
        return self.x[-1] - self.h(len(self.x)) * df(self.x[-1])



class AcceleratedNesterovGradientDescent(MethodOfOptimization):
    def __init__(self, precision=1e-5, convergence_condition=Convergence.ByArgument, h=0.5):
        super().__init__(precision, convergence_condition)
        self.h = h if callable(h) else lambda step: h
        self.y = []


    def pre_call(self, f, x0, df, d2f, g, dg):
        super().pre_call(f, x0, df, d2f, g, dg)
        del self.y[:]
        self.y.append(x0)

    def next_x(self, f, df, d2f, g, dg):
        return self.y[-1] - self.h(len(self.x)) * df(self.y[-1])

    def update(self, f, df, d2f, g, dg):
        self.y.append(self.x[-1] + (len(self.x) / (len(self.x) + 3.)) * (self.x[-1] - self.x[-2]))



class PenaltyMethod(MethodOfOptimization):
    def __init__(self, precision=1e-5, convergence_condition=Convergence.ByArgument, 
                       penalty=lambda step, size: ((step - 1) ** 0.7) * np.ones(size),                           # TODO 
                       exterior_penalty_function=lambda x: np.clip(x, a_min=0, a_max=np.inf) ** 2,
                       gamma=lambda x: np.abs(x),
                       unconditional_method=GradientDescent(precision=5e-4, 
                                                            convergence_condition=Convergence.ByArgumentLastThree, 
                                                            h=0.01)):
        super().__init__(precision, convergence_condition)
        self.penalty = penalty
        self.exterior_penalty_function = exterior_penalty_function
        self.gamma = gamma
        self.unconditional_method = unconditional_method


    def next_x(self, f, df, d2f, g, dg):
        return self.unconditional_method(
                f=lambda x: f(x) + np.dot(self.penalty(len(self.x), self.x[-1].shape[0]), 
                                          self.exterior_penalty_function(g(x))), 
                x0=self.x[-1], 
                df=lambda x: df(x) + np.dot(self.penalty(len(self.x), self.x[-1].shape[0]), 
                                            (2 * g(x) * dg(x)).clip(min=0)),    # TODO
                d2f=d2f)
