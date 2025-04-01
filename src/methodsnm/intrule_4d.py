"""
This module provides classes for numerical integration rules in 2D (triangles).
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.intrule import IntRule

class IntRuleTesserakt(IntRule):
    """
    Abstract base class for numerical integration rules on Tesserakt.
    """
    def __init__(self):
        pass

class EdgeMidPointRule4D(IntRuleTesserakt):
    """
    Class for the midpoint rule for 4D numerical integration.
    """
    def __init__(self):
        """
        Initializes the midpoint rule with the given interval.

        """
        self.nodes = array([[0.5,0.5,0.5,0.5]])
        self.weights = array([1.0])
        self.exactness_degree = 1


import numpy as np
from numpy.polynomial.legendre import leggauss
from itertools import product
class IntRule4D(IntRuleTesserakt):
    @staticmethod
    def get_1d_rule_unit_interval(order):
        nodes, weights = leggauss(order)
        # transform from [-1,1] to [0,1]
        nodes = 0.5 * (nodes + 1)
        weights = 0.5 * weights
        return nodes, weights
    @staticmethod
    def get_4d_integration_rule(order):
        """
        Liefert die Knoten und Gewichte einer Tensorprodukt-Integrationsregel in 4D auf [0,1]^4

        Returns:
            nodes_4d: ndarray shape (n^4, 4)
            weights_4d: ndarray shape (n^4,)
        """
        nodes_1d, weights_1d = IntRule4D.get_1d_rule_unit_interval(order)

        # alle Kombinationen von 4D-Punkten
        nodes_4d = []
        weights_4d = []

        for x, y, z, t in product(range(order), repeat=4):
            point = [nodes_1d[x], nodes_1d[y], nodes_1d[z], nodes_1d[t]]
            weight = weights_1d[x] * weights_1d[y] * weights_1d[z] * weights_1d[t]
            nodes_4d.append(point)
            weights_4d.append(weight)

        return np.array(nodes_4d), np.array(weights_4d)
    
    def __init__(self, order):
        """
        Initializes the integration rule with the given order.

        Parameters:
        order (int): The order of the integration rule.
        """

        self.nodes, self.weights = IntRule4D.get_4d_integration_rule(order)
        
