"""
This module provides classes for numerical integration rules in 2D (triangles).
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.intrule import IntRule

class IntRuleTriangle(IntRule):
    """
    Abstract base class for numerical integration rules on triangle.
    """
    def __init__(self):
        pass

class EdgeMidPointRule(IntRuleTriangle):
    """
    Class for the midpoint rule for 1D numerical integration.
    """
    def __init__(self):
        """
        Initializes the midpoint rule with the given interval.

        """
        self.nodes = array([[0.5,0.0],[0.5,0.5],[0.0,0.5]])
        self.weights = array([1.0/6.0,1.0/6.0,1.0/6.0])
        self.exactness_degree = 1

from methodsnm.intrule_1d import NP_GaussLegendreRule
from methodsnm.intrule_1d import SP_GaussJacobiRule
class DuffyBasedRule(IntRule):
    def __init__(self, order):
        gp_points = max(order,0)//2+1
        self.gauss = NP_GaussLegendreRule(gp_points)
        self.gjacobi = SP_GaussJacobiRule(gp_points,alpha=1,beta=0)
        self.nodes = array([[(1-eta[0])*xi[0], eta[0]] for xi in self.gauss.nodes for eta in self.gjacobi.nodes])
        self.weights = array([w1*w2 for w1 in self.gauss.weights for w2 in self.gjacobi.weights])
        self.exactness_degree = 2*gp_points-1