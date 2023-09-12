from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe import *

class FE_2D(FE):
    num_diff_warned = False

    def __init__(self):
        self.dim = 12

    def _evaluate_id(self, ip):
        raise Exception("Not implemented - Base class should not be used")

    def _evaluate_deriv(self, ip):
        # numerical differentiation - should be overwritten by subclasses
        # for proper accuracy and performance
        if not FE_2D.num_diff_warned:
            print("Warning: Using numerical differentiation for deriv evaluation in " + str(type(self)) + " object.")
            FE_2D.num_diff_warned = True
        eps = 1e-8
        left = ip.copy() - array([eps,0])
        right = ip.copy() + array([eps,0])
        top = ip.copy() + array([0,eps])
        bottom = ip.copy() - array([0,eps])
        return array([[(self._evaluate_id(right) - self._evaluate_id(left))/(2*eps)], 
                      [(self._evaluate_id(top) - self._evaluate_id(bottom))/(2*eps)]])


class TriangleFE(FE_2D):

    def __init__(self):
        super().__init__()
        self.eltype = "triangle"

    @abstractmethod
    def _evaluate_id(self, ip):
        raise Exception("Not implemented - Base class should not be used")

class P1_Triangle_FE(TriangleFE, Lagrange_FE):
    """
    This class represents a P1 triangle finite element.
    """
    ndof = 3
    order = 1
    def __init__(self):
        super().__init__()
        self.nodes = [ np.array([0,0]), np.array([1,0]), np.array([0,1]) ]

    def _evaluate_id(self, ip):
        """
        Evaluates the P1 triangle finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the P1 triangle finite element at the given integration point.
        """
        return array([1-ip[0]-ip[1], ip[0], ip[1]])

    def __str__(self):
        return "P1 Triangle Finite Element"


class P2_Triangle_FE(TriangleFE, Lagrange_FE):
    """
    This class represents a P2 triangle finite element.
    """
    ndof = 6
    order = 2
    def __init__(self):
        super().__init__()
        self.nodes = [ np.array([0,0]), np.array([1,0]), np.array([0,1]),
                       np.array([0.5,0.5]), np.array([0,0.5]), np.array([0.5,0]) ]

    def _evaluate_id(self, ip):
        """
        Evaluates the P2 triangle finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the P2 triangle finite element at the given integration point.
        """
        raise Exception("Not implemented")

    def __str__(self):
        return "P2 Triangle Finite Element\n" + super().__str__()

