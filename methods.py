import numpy.linalg as la
from enum import Enum
import numpy as np
from cvxopt import matrix, solvers


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
            x = self.next_x(f, df, d2f, g, dg)
            if any(np.isnan(x)) or any(np.isinf(x)) or np.isnan(f(x)) or np.isinf(f(x)):
                return  self.x[-1]
            self.x.append(x)
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
                       penalty=lambda step, size: ((step - 1) ** 0.7) * np.ones(size),
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
                                            (2 * g(x) @ dg(x)).clip(min=0)),
                d2f=d2f)


class NewtonMethod(MethodOfOptimization):
    def __init__(self, precision=1e-5, convergence_condition=Convergence.ByArgument):
        super().__init__(precision, convergence_condition)

    def next_x(self, f, df, d2f, g, dg):
        d2 = d2f(self.x[-1])
        if isinstance(d2, (float, int)):
            d2 = np.array([d2])
        inv = la.inv(d2) if len(d2.shape) >= 2 else 1. / d2
        print(inv, self.x[-1])
        return self.x[-1] - np.dot(inv, df(self.x[-1]))


class QuasiNewtonMethod(MethodOfOptimization):
    def __init__(self, precision=1e-5, convergence_condition=Convergence.ByArgument):
        super().__init__(precision, convergence_condition)
        self.H = []

    def pre_call(self, f, x0, df, d2f, g, dg):
        super().pre_call(f, x0, df, d2f, g, dg)
        del self.H[:]
        self.H.append(np.ones((x0.shape[0], x0.shape[0])))

    def next_x(self, f, df, d2f, g, dg):
        return self.x[-1] - (self.H[-1] @ df(self.x[-1]).T)


class BroydenFletcherGoldfarbShannoMethod(QuasiNewtonMethod):
    def __init__(self, precision=1e-5, convergence_condition=Convergence.ByArgument):
        super().__init__(precision, convergence_condition)

    def update(self, f, df, d2f, g, dg):
        E = np.ones(self.H[-1].shape)
        s = np.array([self.x[-1] - self.x[-2]])
        y = np.array([df(self.x[-1]) - df(self.x[-2])])
        self.H.append((E - (s.T @ y) / (y[0] @ s[0])) @ self.H[-1] @ (E - (y.T @ s) / (y[0] @ s[0])) + (s.T @ s) / (y[0] @ s[0]))


class ConditionalGradientMethod(MethodOfOptimization):
    def __init__(self, precision=1e-5, convergence_condition=Convergence.ByValue,
                       k=lambda step: 2. / (step + 3.)):
        super().__init__(precision, convergence_condition)
        self.k = k if callable(k) else lambda step: k
    

    def next_x(self, f, df, d2f, g, dg):
        c = matrix(df(self.x[-1]))
        b = matrix(-g(np.zeros(self.x[-1].shape[0])))

        def basis(i):
            tmp = np.zeros(self.x[-1].shape[0])
            tmp[i] = 1.
            return tmp

        A = matrix([matrix(g(basis(i)) + b.T) for i in range(self.x[-1].shape[0])]).T

        s = np.array(solvers.lp(c, A, b, options={'show_progress' : False})['x']).T[0]

        return self.x[-1] + self.k(len(self.x)) * (s - self.x[-1])
