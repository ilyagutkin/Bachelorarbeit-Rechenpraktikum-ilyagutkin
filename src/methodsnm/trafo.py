from abc import ABC, abstractmethod
import numpy as np
from numpy import array

from methodsnm.mesh import Mesh

class Transformation(ABC):
    dim_domain = None
    dim_range = None
    eltype = None

    def __init__(self, interval):
        """Abstract base for mappings from reference -> physical element.

        Subclasses provide ``_map`` and ``_jacobian`` for single point
        evaluation. Public helpers ``map`` and ``jacobian`` accept
        either a single point or an array of points.
        """
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
        """Base class for element-local transformations.

        Holds a reference to the parent mesh and the local element
        index (``elnr``). Concrete element transformations inherit from
        this class.
        """

class IntervalTransformation(ElementTransformation):
    """Affine mapping from reference interval [0,1] to a physical segment.

    Maps xi in [0,1] to x = a + (b-a)*xi where [a,b] are the element
    endpoints stored in the mesh.
    """
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

    """Affine mapping from reference triangle to a physical triangle.

    The transformation is x = a + (b-a)*xi + (c-a)*eta and the
    Jacobian is constant for affine elements.
    """

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

    """Reference-to-physical mapping for structured tesseract elements.

    Uses a simple scaling by the grid spacing in each coordinate; the
    Jacobian is diagonal and constant for the structured tesseract.
    """
    
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
        a, b, c, d, e = self.points
        # Jacobi-Matrix: Spaltenvektoren von (b-a), (c-a), (d-a), (e-a)
        return np.column_stack([b - a, c - a, d - a, e - a])  # Shape (4, 4)
    """Affine mapping from reference 4D simplex to a physical hypertriangle.

    The reference coordinates (eps1..eps4) are mapped affinely using
    the 5 vertex positions. Jacobian is constant for affine elements.
    """
    
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

    def _map(self, ip):
        """Map reference simplex coordinates to physical 4D point."""
        a, b, c, d, e = self.points
        eps1, eps2, eps3, eps4 = ip
        return a + (b - a)*eps1 + (c - a)*eps2 + (d - a)*eps3 + (e - a)*eps4

    def _jacobian(self, ip):
        """Return the (constant) Jacobian matrix for this affine map."""
        return self.jac