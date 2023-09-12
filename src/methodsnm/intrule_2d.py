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
        
