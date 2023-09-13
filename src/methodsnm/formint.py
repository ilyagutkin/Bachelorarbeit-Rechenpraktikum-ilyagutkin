from abc import ABC, abstractmethod
import numpy as np
from numpy import array, einsum
from methodsnm.intrule import *
from methodsnm.fe import *
from numpy.linalg import det, inv
from methodsnm.meshfct import ConstantFunction

class FormIntegral(ABC):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Not implemented")

class LinearFormIntegral(FormIntegral):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Not implemented")

    def compute_element_vector(self, fe, trafo):
        raise NotImplementedError("Not implemented")

class SourceIntegral(LinearFormIntegral):

    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_vector(self, fe, trafo, intrule = None):
        raise NotImplementedError("Not implemented")

class BilinearFormIntegral(FormIntegral):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Not implemented")

    def compute_element_matrix(self, fe, trafo):
        raise NotImplementedError("Not implemented")

class MassIntegral(BilinearFormIntegral):

    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule = None):
        raise NotImplementedError("Not implemented")

class LaplaceIntegral(BilinearFormIntegral):

    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule = None):
        raise NotImplementedError("Not implemented")

