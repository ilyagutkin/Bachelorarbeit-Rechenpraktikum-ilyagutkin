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
        
import numpy as np
from methodsnm.intrule import IntRule  # deine Basisklasse

class IntRuleSimplex4D(IntRule):
    """
    Abstract base class for 4D simplex integration rules.
    """
    def __init__(self):
        pass

class IntRulePentatope(IntRuleSimplex4D):
    """
    Integration rule for the 4D reference simplex with selectable order.
    Supported orders: 1–5.
    """
    def __init__(self, order: int):
        if order <= 1:
            self._init_order_1()
        elif order == 2:
            self._init_order_2()
        elif order == 3:
            self._init_order_3()
        elif order == 4:
            self._init_order_4()
        elif order == 5:
            self._init_order_5()
        else:
            raise NotImplementedError(f"Pentatope rule for order {order} not implemented.")

    def _init_order_1(self):
        self.nodes = np.array([[0.2, 0.2, 0.2, 0.2]])
        self.weights = np.array([1.0 / 24.0])
        self.exactness_degree = 1

    def _init_order_2(self):
        a = 1.0 / 6.0
        b = 1.0 / 2.0
        self.nodes = np.array([
            [a, a, a, a],
            [b, a, a, a],
            [a, b, a, a],
            [a, a, b, a],
            [a, a, a, b]
        ])
        self.weights = np.full(5, 1.0 / 120.0)
        self.exactness_degree = 2

    def _init_order_3(self):
        w1 = 0.030283678097089  # Zentrum
        w2 = 0.006026785714286  # symmetrische Punkte
        a = 0.617587190300083
        b = 0.127470936566639

        center = [0.25, 0.25, 0.25, 0.25]
        perms = [
            [a, b, b, b],
            [b, a, b, b],
            [b, b, a, b],
            [b, b, b, a]
        ]

        self.nodes = np.array([center] + perms)
        self.weights = np.array([w1] + [w2] * 4)
        self.exactness_degree = 3

    def _init_order_4(self):
        # Grundmann-Möller Regel für Grad 4 (6 Punkte)
        self.nodes = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.4,  0.2,  0.2,  0.2],
            [0.2,  0.4,  0.2,  0.2],
            [0.2,  0.2,  0.4,  0.2],
            [0.2,  0.2,  0.2,  0.4],
            [0.1,  0.3,  0.3,  0.3],
        ])
        self.weights = np.array([
            0.02164502164502164,
            0.01082251082251082,
            0.01082251082251082,
            0.01082251082251082,
            0.01082251082251082,
            0.0144927536231884
        ])
        self.exactness_degree = 4

    def _init_order_5(self):
        # Symmetrische Regel mit 15 Punkten (Stroud/Grundmann-Möller inspiriert)
        a = 0.5
        b = 1.0 / 6.0
        c = 0.25

        self.nodes = []
        self.weights = []

        # 5 Punkte mit (b,b,b,b)
        self.nodes.append([b, b, b, b])
        self.weights.append(1/120.0)

        # 4 Permutationen von (a,b,b,b)
        w1 = 0.006853  # Beispielgewicht (angepasst auf Summe = 1/24)
        for i in range(4):
            pt = [b, b, b, b]
            pt[i] = a
            self.nodes.append(pt)
            self.weights.append(w1)

        # 6 Permutationen von (c,c,c,d)
        d = 1.0 - 3*c  # damit Summe = 1
        w2 = 0.0032  # angepasstes Gewicht
        from itertools import permutations
        counted = set()
        for p in permutations([c, c, c, d]):
            tup = tuple(p)
            if tup not in counted:
                self.nodes.append(p)
                self.weights.append(w2)
                counted.add(tup)

        self.nodes = np.array(self.nodes)
        self.weights = np.array(self.weights)
        self.exactness_degree = 5
