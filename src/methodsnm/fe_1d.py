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

class P2_Segment_FE(FE_1D):
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
        Evaluates the P2 segment finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the P2 segment finite element at the given integration point.
        """
        return array([1-ip[0], ip[0], 4*ip[0]*(1-ip[0])])

    def __str__(self):
        return "P2 Segment Finite Element\n" + super().__str__()
    
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
        self.barycentric_weights = np.ones(self.ndof)
        for i in range(self.ndof):
            for j in range(self.ndof):
                if i != j:
                    self.barycentric_weights[i] /= (self.nodes[i] - self.nodes[j])              

    def _evaluate_id(self, ip):
        """
        Evaluates the Lagrange segment finite element at the given integration point.

        Uses the barycentric form of the Lagrange polynomials, 
        see https://en.wikipedia.org/wiki/Lagrange_polynomial#Barycentric_form

        l_j(x) = prod_{i!=j} (x-x_i)/(x_j-x_i) 
               = w_j * prod_{i!=j} (x-x_i) with w_j = prod_i (1/(x_j-x_i))
               = w_j / (x-x_j) * l(x) with l(x) = prod_i (x-x_i)
        With further
             1 = sum_i l_i(x) = sum_i (w_i / (x-x_i)) * l(x) 
               = l(x) * sum_i (w_i / (x-x_i))
        we have
        l_j(x) = w_j / (x-x_j) * sum_i (w_i / (x-x_i))
        where the last sum is a does not depend on j.

        Evaluation costs are hence O(ndof) instead of O(ndof^2) for the naive approach.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the Lagrange segment finite element at the given integration point.
        """
        if ip[0] in self.nodes:
            ret = np.zeros(self.ndof)
            ret[self.nodes.index(ip[0])] = 1
        else:
            denom = sum([self.barycentric_weights[i]/(ip[0]-self.nodes[i]) for i in range(self.ndof)])
            ret = self.barycentric_weights.copy()
            for i in range(self.ndof):
                ret[i] /= (ip[0]-self.nodes[i]) * denom
        return ret

    def __str__(self):
        return f"Lagrange Segment Finite Element(order={self.order})\n" + super().__str__()

from methodsnm.recpol import *
class RecPol_Segment_FE(FE_1D):
    """
    This class represents a Recursive Polynomial finite element on [0,1].
    """
    def __init__(self, order, recpol):
        super().__init__()
        self.order = order
        self.ndof = order+1
        self.recpol = recpol

    def _evaluate_id(self, ip):
        return self.recpol.evaluate_all(2*ip-1, self.order)

    def _evaluate_id_array(self, ip):
        return self.recpol.evaluate_all(2*ip[:,0]-1, self.order)

    def __str__(self):
        return f"RecPol Segment Finite Element(recpol={self.recpol}, order={self.order})\n" + super().__str__()

def Legendre_Segment_FE(order):
    return RecPol_Segment_FE(order, LegendrePolynomials())

def Jacobi_Segment_FE(order, alpha, beta):
    return RecPol_Segment_FE(order, JacobiPolynomials(alpha, beta))
