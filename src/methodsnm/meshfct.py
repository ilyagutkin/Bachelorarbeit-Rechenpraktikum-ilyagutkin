from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.trafo import *

class MeshFunction:
    mesh = None
    def __init__(self, mesh):
        self.mesh = mesh
        pass

    @abstractmethod
    def _evaluate(self, ip, trafo):
        raise Exception("Not implemented - Base class should not be used")

    def _evaluate_array(self, ips, trafo):
        ret = np.empty(ips.shape[0])
        for i in range(ips.shape[0]):
            ret[i] = self.evaluate(ips[i], trafo)
        return ret       

    def evaluate(self, ip, trafo):
        if isinstance(ip, np.ndarray):
            if ip.ndim == 1:
                return self._evaluate(ip, trafo)
            else:
                return self._evaluate_array(ip, trafo)
        else:
            raise Exception("Invalid input")


class ConstantFunction(MeshFunction):
    c = None
    def __init__(self, c, mesh=None):
        self.mesh = mesh
        self.c = c

    def _evaluate(self, ip, trafo):
        return self.c

class GlobalFunction(MeshFunction):
    f = None
    def __init__(self, function, mesh):
        self.mesh = mesh
        self.f = function

    def _evaluate(self, ip, trafo):
        return self.f(trafo(ip))

class FEFunction(MeshFunction):
    fes = None
    vector = None
    def __init__(self, fes):
        self.mesh = fes.mesh
        self.fes = fes
        self.vector = np.zeros(fes.ndof)

    def _evaluate(self, ip, trafo= None):
        if trafo is None:
            el = self.mesh.find_element(ip)
            trafo = self.mesh.trafo(el)
        fe = self.fes.finite_element(trafo.elnr)
        dofs = self.fes.element_dofs(trafo.elnr)
        return np.dot(fe.evaluate(ip), self.vector[dofs])

    def _evaluate_array(self, ips, trafo):
        fe = self.fes.finite_element(trafo.elnr)
        dofs = self.fes.element_dofs(trafo.elnr)
        return np.dot(fe.evaluate(ips), self.vector[dofs])

    def _set(self, f , boundary=False):
        """
        Sets the values of the finite element function to zero.
        """
        if str(self.fes.fe) != "P1 Triangle Finite Element":
            raise Exception("Only P1 triangle finite element is supported")
        if boundary:
            for dof in self.mesh.bndry_vertices:
                x = self.mesh.points[dof]
                self.vector[dof] = f(x)
        else:
            for dof in self.mesh.vertices:
                x = self.mesh.points[dof]
                self.vector[dof] = f(x)
            