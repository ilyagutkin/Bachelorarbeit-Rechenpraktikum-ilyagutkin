from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe import *

class FE_1D(FE):
    """
    Abstract base class for finite elements in 1D.
    It implements a derivative evaluation using numerical differentiation.
    """    
    num_diff_warned = False

    def __init__(self):
        self.eltype = "segment"
        self.dim = 1

    @abstractmethod
    def _evaluate_id(self, ip):
        raise Exception("Not implemented - Base class should not be used")

    def _evaluate_deriv(self, ip):
        # numerical differentiation - should be overwritten by subclasses
        # for proper accuracy and performance
        if not FE_1D.num_diff_warned:
            print("Warning: Using numerical differentiation for deriv evaluation in " + str(type(self)) + " object.")
            FE_1D.num_diff_warned = True
        eps = 1e-8
        left = ip.copy() - eps
        right = ip.copy() + eps
        return ((self._evaluate_id(right) - self._evaluate_id(left))/(2*eps)).reshape((1,self.ndof))

class P1_Segment_FE(FE_1D, Lagrange_FE):
    """
    This class represents a P1 segment finite element.
    """
    ndof = 2
    order = 1
    def __init__(self):
        super().__init__()
        self.nodes = [ np.array([0]), np.array([1]) ]

    def _evaluate_id(self, ip):
        """
        Evaluates the P1 segment finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the P1 segment finite element at the given integration point.
        """
        return array([1-ip[0], ip[0]])

    def __str__(self):
        return "P1 Segment Finite Element\n" + super().__str__()

class P1Mod_Segment_FE(FE_1D, Lagrange_FE):
    """
    This class represents a P1 segment finite element.
    """
    ndof = 3
    order = 2
    def __init__(self):
        super().__init__()
        self.nodes = [ np.array([0]), np.array([1]) ]

    def _evaluate_id(self, ip):
        """
        Evaluates the P1 segment finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the P1 segment finite element at the given integration point.
        """
        phi3 = -6*((ip[0])**2 - ip[0])
        return array([1-ip[0], ip[0], phi3])

    def __str__(self):
        return "P1 Segment Finite Element\n" + super().__str__()
    
class Lagrange_Segment_FE(Lagrange_FE, FE_1D):
    """
    This class represents a Lagrange finite element on [0,1].
    """
    def __init__(self, order, nodes=None):
        super().__init__()
        self.order = order
        self.ndof = order+1
        if nodes is not None:
            if len(nodes) != self.ndof:
                raise Exception("Invalid number of nodes")
            self.nodes = nodes
        else:
            self.nodes = [ np.array(x) for x in np.linspace(0, 1, self.ndof) ]

    def _evaluate_id(self, ip):
        """
        Evaluates the Lagrange segment finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the Lagrange segment finite element at the given integration point.
        """
        raise Exception("Not implemented")

    def __str__(self):
        return f"Lagrange Segment Finite Element(order={self.order})\n" + super().__str__()
