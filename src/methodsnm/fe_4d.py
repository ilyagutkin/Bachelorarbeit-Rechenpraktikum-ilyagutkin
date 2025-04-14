from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe import *

class FE_4D(FE):
    """
    Abstract base class for finite elements in 4D.
    It implements a derivative evaluation using numerical differentiation.
    """    

    num_diff_warned = False

    def __init__(self):
        self.dim = 4

    def _evaluate_id(self, ip):
        raise Exception("Not implemented - Base class should not be used")

    def _evaluate_deriv(self, ip):
        # numerical differentiation - should be overwritten by subclasses
        # for proper accuracy and performance
        if not FE_4D.num_diff_warned:
            print("Warning: Using numerical differentiation for deriv evaluation in " + str(type(self)) + " object.")
            FE_4D.num_diff_warned = True
        eps = 1e-8
        left = ip.copy() - array([eps,0,0,0])
        right = ip.copy() + array([eps,0,0,0])
        left2= ip.copy() - array([0,eps,0,0])
        right2 = ip.copy() + array([0,eps,0,0])
        left3= ip.copy() - array([0,0,eps,0])
        right3 = ip.copy() + array([0,0,eps,0])
        left4= ip.copy() - array([0,0,0,eps])
        right4 = ip.copy() + array([0,0,0,eps])
        return array([(self._evaluate_id(right) - self._evaluate_id(left))/(2*eps), 
                      (self._evaluate_id(right2) - self._evaluate_id(left2))/(2*eps),
                      (self._evaluate_id(right3) - self._evaluate_id(left3))/(2*eps),
                      (self._evaluate_id(right4) - self._evaluate_id(left4))/(2*eps)])


class TesseraktFE(FE_4D):
    """
    Abstract base class for finite elements on tesserakt.
    """    

    def __init__(self):
        super().__init__()
        self.eltype = "tesserakt"

    @abstractmethod
    def _evaluate_id(self, ip):
        raise Exception("Not implemented - Base class should not be used")
class P1_Tesserakt_FE(TesseraktFE,Lagrange_FE):
    def _evaluate_deriv(self, ip):
        return super()._evaluate_deriv(ip)
    """
    This class represents a P1 Tesserakt finite element.
    """
    ndof = 2**4
    order = 1
    def __init__(self):
        super().__init__()
        self.nodes = [np.array([x, y, z, w]) for x in [0, 1] for y in [0, 1] for z in [0, 1] for w in [0, 1]]

    def _evaluate_id(self, ip):
        """
        Evaluates the P1 Tesserakt finite element at the given integration point.

        Parameters:
        ip (numpy.ndarray): The integration point at which to evaluate the finite element.

        Returns:
        numpy.ndarray: The values of the P1 Tesserakt finite element at the given integration point.
        """

        x, y, z, w = ip
        return np.array([
            (1-x)*(1-y)*(1-z)*(1-w),
                x*(1-y)*(1-z)*(1-w),
            (1-x)*    y*(1-z)*(1-w),
                x*    y*(1-z)*(1-w),
            (1-x)*(1-y)*    z*(1-w),
                x*(1-y)*    z*(1-w),
            (1-x)*    y*    z*(1-w),
                x*    y*    z*(1-w),
            (1-x)*(1-y)*(1-z)*   w,
                x*(1-y)*(1-z)*   w,
            (1-x)*    y*(1-z)*   w,
                x*    y*(1-z)*   w,
            (1-x)*(1-y)*    z*   w,
                x*(1-y)*    z*   w,
            (1-x)*    y*    z*   w,
                x*    y*    z*   w
        ])


    def __str__(self):
        return "P1 Tesserakt Finite Element"