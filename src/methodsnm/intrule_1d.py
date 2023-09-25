"""
This module provides classes for 1D numerical integration rules.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.intrule import IntRule

class IntRule1D(IntRule):
    """
    Abstract base class for 1D numerical integration rules.
    """
    interval = None
    def __init__(self, interval=(0,1)):
        """
        Initializes the integration rule with the given interval.
        """
        self.interval = interval


def evaluate_exactness_degree(rule, max_order=None):
    """
    Evaluates the exactness degree of the given integration rule.

    Parameters:
    rule (IntRule1D): The integration rule to evaluate.
    max_order (int or None): maximum order to check for exactness degree. If None, the exactness degree is checked until it is found.

    Returns:
    int: The exactness degree of the integration rule.
    """
    a,b = rule.interval
    i=0
    while True:
        if not np.isclose(rule.integrate(lambda x: x**i), (b**(i+1)-a**(i+1))/(i+1), rtol=1e-12):
            if i == 0:
                print("Warning: Exactness degree is below 0")
            return i-1
        i += 1
        if max_order is not None and i > max_order:
            raise ValueError("Could not determine exactness degree")

class MidPointRule(IntRule1D):
    """
    Class for the midpoint rule for 1D numerical integration.
    """
    def __init__(self, interval = (0,1)):
        """
        Initializes the midpoint rule with the given interval.

        Parameters:
        interval (tuple): The interval to integrate over.
        """
        self.interval = interval
        a,b = interval
        self.nodes = array([[0.5*a+0.5*b]])
        self.weights = array([1.0*(b-a)])
        self.exactness_degree = 1


class NewtonCotesRule(IntRule1D):
    """
    Class for the Newton-Cotes rule for 1D numerical integration.
    """
    def __init__(self, n =None, nodes=None, interval=(0,1)):
        """
        Initializes the Newton-Cotes rule with the given interval and number of nodes.

        Parameters:
        n (int or list): The number of nodes or a list of nodes to use for integration.
        nodes (list): A list of nodes to use for integration.
        interval (tuple): The interval to integrate over.
        """
        self.interval = interval
        a,b = interval
        if nodes is None and n is None:
            raise ValueError("Either n or nodes must be specified")
        if isinstance(n, list):
            nodes = n
        else:
            nodes = np.linspace(a,b,n)
        n = len(nodes)

        self.nodes = array(nodes)

        h = b-a
        # Compute weights on [0,1]
        x = [(nodes[i]-a)/h for i in range(n)]
        # sum w[i] * x[i]**k = 1/(k+1)
        A = np.array([[x[i]**k for i in range(n)] for k in range(n)])
        b = np.array([1.0/(k+1) for k in range(n)])
        w = np.linalg.solve(A,b)
        self.weights = w*h
        self.exactness_degree = evaluate_exactness_degree(self)

        self.exactness_degree = evaluate_exactness_degree(self)

class NP_GaussLegendreRule(IntRule1D):
    """
    Wrapper class for the Gauss-Legendre rule for 1D numerical integration of numpy.
    """
    def __init__(self, n, interval=(0,1)):
        """
        Initializes the Gauss-Legendre rule with the given interval and number of nodes.

        Parameters:
        n (int): The number of nodes to use for integration.
        interval (tuple): The interval to integrate over.
        """
        self.interval = interval
        a,b = interval
        self.nodes = np.empty((n,1))
        nodes = np.polynomial.legendre.leggauss(n)[0]
        self.nodes[:,0] = 0.5*(a+b) + 0.5*(b-a)*nodes
        self.weights = np.polynomial.legendre.leggauss(n)[1]
        self.weights = 0.5*(b-a)*self.weights
        self.exactness_degree = 2*n-1

class GaussLegendreRule(IntRule1D):
    """
    Class for the Gauss-Legendre rule for 1D numerical integration.
    """
    def __init__(self, n, interval=(0,1)):
        """
        Initializes the Gauss-Legendre rule with the given interval and number of nodes.

        Parameters:
        n (int): The number of nodes to use for integration.
        interval (tuple): The interval to integrate over.
        """
        self.interval = interval
        a,b = interval

        raise NotImplementedError("Not implemented")        

import scipy
class SP_GaussJacobiRule(IntRule1D):
    """
    Wrapper class for the Gauss-Jacobi rule for 1D numerical integration of numpy.
    """
    def __init__(self, n, alpha, beta, interval=(0,1)):
        """
        Initializes the Gauss-Legendre rule with the given interval and number of nodes.

        Parameters:
        n (int): The number of nodes to use for integration.
        interval (tuple): The interval to integrate over.
        alpha, beta (float): The parameters of the Jacobi polynomial.
        """
        self.interval = interval
        a,b = interval
        self.alpha = alpha
        self.beta = beta
        nodes, weights = scipy.special.roots_jacobi(n,alpha,beta)
        self.nodes = np.empty((n,1))
        self.nodes[:,0] = 0.5*(a+b) + 0.5*(b-a)*nodes
        self.weights = 0.5**(alpha+beta+1)*(b-a)*weights
        self.exactness_degree = None

class GaussJacobiRule(IntRule1D):
    """
    Class for the Gauss-Jacobi rule for 1D numerical integration.
    """
    def __init__(self, n, alpha, beta,  interval=(0,1)):
        """
        Initializes the Gauss-Jacobi rule with the given interval and number of nodes.

        Parameters:
        n (int): The number of nodes to use for integration.
        interval (tuple): The interval to integrate over.
        alpha, beta (float): The parameters of the Jacobi polynomial.
        """
        self.interval = interval
        a,b = interval
        self.alpha = alpha
        self.beta = beta

        raise NotImplementedError("Not implemented")                

