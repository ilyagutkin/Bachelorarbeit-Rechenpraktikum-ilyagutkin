from abc import ABC, abstractmethod
import numpy as np
from numpy import array

from methodsnm.mesh import Mesh

class Transformation(ABC):
    dim_domain = None
    dim_range = None
    eltype = None

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
        self.eltype = "segment"

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
        self.eltype = "triangle"

    def _map(self, ip):
        a,b,c = self.points
        return a+(b-a)*ip[0]+(c-a)*ip[1]

    def _jacobian(self, ip):
        return self.jac
    
class TesseraktTransformation(ElementTransformation):
    points = None
    def calculate_jacobian(self):
        M = self.mesh.xlength
        N = self.mesh.ylength
        K = self.mesh.zlength
        L = self.mesh.tlength
        return np.diag([1/M,1/N,1/K,1/L]).T
    
    def __init__(self, mesh_or_points, elnr=None):
        if isinstance(mesh_or_points, Mesh):
            mesh = mesh_or_points
            if elnr is None:
                raise ValueError("TesseraktTransformation needs an element number")
            super().__init__(mesh, elnr)
            self.points = tuple(mesh.points[mesh.elements()[elnr]])
        else:
            super().__init__(mesh = None, elnr = -1)
            self.points = mesh_or_points
        self.jac = self.calculate_jacobian()
        self.dim_range = 4
        self.dim_domain = 4
        self.eltype = "tesserakt"
        
    def _map(self, ip):
        def elementwise_min_point(points):
            return np.min(points, axis=0)

        xmin = elementwise_min_point(self.points)
        return xmin + np.array([ip[0],ip[1],ip[2],ip[3]])*np.array([1/self.mesh.xlength,1/self.mesh.ylength,1/self.mesh.zlength,1/self.mesh.tlength])

    def _jacobian(self,ip):
        return self.jac

class HypertriangleTransformation(ElementTransformation):
    points = None
    def calculate_jacobian(self):
        a,b,c = self.points
        return array([b-a,c-a]).T

    def __init__(self, mesh_or_points, elnr=None):
        if isinstance(mesh_or_points, Mesh):
            mesh = mesh_or_points
            if elnr is None:
                raise ValueError("HypertriangleTransformation needs an element number")
            super().__init__(mesh, elnr)
            self.points = tuple(mesh.points[mesh.elements()[elnr]])
        else:
            super().__init__(mesh = None, elnr = -1)
            self.points = mesh_or_points
        self.jac = self.calculate_jacobian()
        self.dim_range = 4
        self.dim_domain = 4
        self.eltype = "hypertriangle"
        
    def calculate_jacobian(self):
        a, b, c, d, e = self.points
        # Jacobi-Matrix: Spaltenvektoren von (b-a), (c-a), (d-a), (e-a)
        return np.column_stack([b - a, c - a, d - a, e - a])  # Shape (4, 4)

    def _map(self, ip):
        """Mappt ip ∈ Referenzsimplex (z. B. baryzentrisch) in echten 4D-Raum"""
        a, b, c, d, e = self.points
        ξ1, ξ2, ξ3, ξ4 = ip
        return a + (b - a)*ξ1 + (c - a)*ξ2 + (d - a)*ξ3 + (e - a)*ξ4

    def _jacobian(self, ip):
        """Affine Transformation ⇒ Jacobi-Matrix ist konstant"""
        return self.jac