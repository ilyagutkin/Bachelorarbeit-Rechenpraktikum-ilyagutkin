from numpy import array
from methodsnm.fe import *

import numpy as np
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


class RecursivePolynomial(ABC):
    """
    Abstract base class for recursive polynomial evaluation.

    Polynomials are evaluated using the following recursion:
        * d "starter" (given functions)
            * P_i(x) = s_i(x) for i < d
        * recursion formulate for i >= d:
            * P_i(x) = sum_{j=0}^{d-1} c_j(i,x) P_{i-d+j}(x)

    Attributes:
    -----------
    starter : list
        List of starter functions.
    rec_coeff : list
        List of recursive coefficients.
    """
    starter = None
    rec_coeff = None

    @abstractmethod
    def __init__(self):
        pass

    def evaluate_all(self, x, n):
        """
        Evaluates the recursive polynomial for all values of x up to n.

        Parameters:
        -----------
        x : array_like
            Array of values to evaluate the polynomial at.
        n : int
            Maximum degree of the polynomial to evaluate.

        Returns:
        --------
        vals : ndarray
            Array of shape (len(x), n+1 ) containing the values of the polynomial
            evaluated at each value of x up to degree n.
        """
        vals = np.empty((len(x), n+1))
        d = len(self.rec_coeff)
        for si, s in enumerate(self.starter):
            vals[:,si] = s(x)
        for i in range(d, n+1):
            vals[:,i] = sum([self.rec_coeff[j](i,x)*vals[:,i-d+j] for j in range(d)])
        return vals

    def evaluate(self, x, n):
        """
        Evaluate the recursive polynomial of degree n for all values of x.

        Parameters:
        -----------
        x : array_like
            Array of values to evaluate the polynomial at.
        n : int
            Degree of the polynomial to evaluate.

        Returns:
        --------
        vals : ndarray
            Array of length len(x) containing the values of the polynomial
            evaluated at each value of x.
        """
        d = len(self.rec_coeff)
        vals = np.empty((len(x), d+1))
        for si, s in enumerate(self.starter):
            vals[:,si] = s(x)
        for i in range(d, n+1):
            if i != d:
                for j in range(d):
                    vals[:,j] = vals[:,j+1]
            vals[:,d] = sum([self.rec_coeff[j](i,x)*vals[:,j] for j in range(d)])
        return vals[:,d]

    def plot_all(self, x, n):
        """
        Plots the recursive polynomial for all values of x up to n.

        Parameters:
        -----------
        x : array_like
            Array of values to evaluate the polynomial at.
        n : int
            Maximum degree of the polynomial to evaluate.
        """
        vals = self.evaluate_all(x, n)
        for i in range(n+1):
            plt.plot(x, vals[:,i], label="P_{}".format(i))
        plt.legend()
        plt.show()


class Monomials(RecursivePolynomial):
    def __init__(self):
        self.starter = [lambda x: np.ones_like(x)]
        self.rec_coeff = [lambda n,x: x]
