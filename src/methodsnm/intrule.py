from abc import ABC, abstractmethod
import numpy as np
from numpy import array

class IntRule:
    nodes = None
    weights = None
    exactness_degree = None
    def __init__(self):
        pass

    def integrate(self, f):
        # f is assumed to be a function that takes a numpy array as input
        fvals = array([f(self.nodes[i]) for i in range(len(self.weights))])
        return np.dot(fvals.T, self.weights)

    def __str__(self):
        return f"Integration rule \"{self.__class__.__name__}\" with {len(self.nodes)} nodes (exactness degree {self.exactness_degree}):\nnodes = {self.nodes}\nweights = {self.weights}"

from methodsnm.intrule_1d import MidPointRule, NewtonCotesRule
from methodsnm.intrule_2d import EdgeMidPointRule

def select_integration_rule(order, eltype):
    if eltype == "segment":
        if order == 1:
            return MidPointRule()
        else:
            return NewtonCotesRule(n=order+1)
    elif eltype == "triangle":
        if order == 1:
            return EdgeMidPointRule()
        else:
            raise NotImplementedError("Not implemented")
    else:
        raise NotImplementedError("select_integration_rule only implemented for segments and triangles (not for " + eltype + ", yet)")

