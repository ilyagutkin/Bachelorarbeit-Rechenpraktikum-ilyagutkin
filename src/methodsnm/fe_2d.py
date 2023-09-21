from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe import *

class FE_2D(FE):
    """
    Abstract base class for finite elements in 2D.
    It implements a derivative evaluation using numerical differentiation.
    """    

    num_diff_warned = False

    def __init__(self):
        self.dim = 2

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
        ret = array([(self._evaluate_id(right) - self._evaluate_id(left))/(2*eps), 
                      (self._evaluate_id(top) - self._evaluate_id(bottom))/(2*eps)])
        return ret


class TriangleFE(FE_2D):
    """
    Abstract base class for finite elements on triangles.
    """    

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
    This class represents a P2 triangle (Lagrange) finite element.
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
        ret = np.empty(self.ndof)

        x,y = ip
        lamb = [1-x-y,x,y]

        for i in range(3):
            ret[i] = lamb[i]*(2*lamb[i]-1)
        for i in range(3):
            for j in range(i):
                ret[6-i-j] = lamb[i]*lamb[j]*4
        return ret

    def __str__(self):
        return "P2 Triangle Finite Element\n" + super().__str__()


class P3_Triangle_FE(TriangleFE, Lagrange_FE):
    """
    This class represents a P3 triangle (Lagrange) finite element.
    (first draft : Warning: we may want to change some things later on!)
    """
    ndof = 10
    order = 3
    def __init__(self):
        super().__init__()
        self.nodes = [ np.array([0,0]), np.array([1,0]), np.array([0,1]),
                       np.array([2/3,1/3]), np.array([1/3,2/3]), np.array([0,1/3]), np.array([0,2/3]), np.array([1/3,0]), np.array([2/3,0]), 
                       np.array([1/3,1/3]) ]

    def _evaluate_id(self, ip):
        """
        Evaluates the P3 triangle finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the P3 triangle finite element at the given integration point.
        """
        ret = np.empty(self.ndof)

        x,y = ip
        lamb = [1-x-y,x,y]

        #vertex dofs:
        for i in range(3):
            ret[i] = 9.0/2.0*lamb[i]*(lamb[i]-1/3)*(lamb[i]-2/3)

        #edge dofs:
        for i in range(1,3): # major vertex 
            for j in range(i): # minor vertex
                edge_number = 3-i-j
                ret[3+2*edge_number] = lamb[i]*lamb[j]*(lamb[j]-1/3)*27/2   # node next to minor vertex
                ret[3+2*edge_number+1] = lamb[i]*lamb[j]*(lamb[i]-1/3)*27/2 # node next to major vertex
        ret[9] = 27*lamb[0]*lamb[1]*lamb[2]
        return ret

    def __str__(self):
        return "P3 Triangle Finite Element\n" + super().__str__()




class P1Edge_Triangle_FE(TriangleFE, Lagrange_FE):
    """
    This class represents a P1 triangle finite element.
    """
    ndof = 3
    order = 1
    def __init__(self):
        super().__init__()
        self.nodes = [ np.array([0.5,0.5]), np.array([0,0.5]), np.array([0.5,0]) ]

    def _evaluate_id(self, ip):
        """
        Evaluates the P1Edge triangle finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the P1Edge triangle finite element at the given integration point.
        """
        x,y = ip
        lamb = [1-x-y,x,y]
        return array([1-2*lamb[i] for i in range(3)])

    def __str__(self):
        return "P1Edge Triangle Finite Element"