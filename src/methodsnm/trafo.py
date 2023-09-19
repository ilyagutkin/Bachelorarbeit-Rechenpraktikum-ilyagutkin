from abc import ABC, abstractmethod
import numpy as np
from numpy import array

from methodsnm.mesh import Mesh

class Transformation(ABC):
    dim_domain = None
    dim_range = None

    def __init__(self, interval):
        pass

    @abstractmethod
    def _map(self, ip):
        raise NotImplementedError("Not implemented")

    def _map_array(self, ips):
        ret = np.empty(ips.shape)
        for i in range(ips.shape[0]):
            ret[i,:] = self._map(ips[i])
        return ret

    @abstractmethod
    def _jacobian(self, ip):
        raise NotImplementedError("Not implemented")

    def _jacobian_array(self, ips):
        ret = np.empty((ips.shape[0], self.dim_range, self.dim_domain))
        for i in range(ips.shape[0]):
            ret[i,:,:] = self._jacobian(ips[i])
        return ret

    def map(self, ip):
        if ip.ndim == 1:
            return self._map(ip)
        else:
            return self._map_array(ip)

    def jacobian(self, ip):
        if ip.ndim == 1:
            return self._jacobian(ip)
        else:
            return self._jacobian_array(ip)

    def __call__(self, ip):
        return self.map(ip)

class ElementTransformation(Transformation):
    mesh = None
    elnr = None

    def __init__(self, mesh, elnr):
        self.elnr = elnr
        self.mesh = mesh

class IntervalTransformation(ElementTransformation):
    # This class represents a transformation from the interval [0,1] to the given interval.
    def __init__(self, mesh, elnr):
        super().__init__(mesh, elnr)
        self.interval = tuple(mesh.points[mesh.elements()[elnr]])
        self.dim_range = 1
        self.dim_domain = 1

    def _map(self, ip):
        a,b = self.interval
        return a+(b-a)*ip[0]

    def _jacobian(self, ip):
        a,b = self.interval
        return np.array([[b-a]])

class TriangleTransformation(ElementTransformation):
    points = None
    def calculate_jacobian(self):
        a,b,c = self.points
        return array([b-a,c-a]).T

    def __init__(self, mesh_or_points, elnr=None):
        if isinstance(mesh_or_points, Mesh):
            mesh = mesh_or_points
            if elnr is None:
                raise ValueError("TriangleTransformation needs an element number")
            super().__init__(mesh, elnr)
            self.points = tuple(mesh.points[mesh.elements()[elnr]])
        else:
            super().__init__(mesh = None, elnr = -1)
            self.points = mesh_or_points
        self.jac = self.calculate_jacobian()
        self.dim_range = 2
        self.dim_domain = 2

    def _map(self, ip):
        a,b,c = self.points
        return a+(b-a)*ip[0]+(c-a)*ip[1]

    def _jacobian(self, ip):
        return self.jac

