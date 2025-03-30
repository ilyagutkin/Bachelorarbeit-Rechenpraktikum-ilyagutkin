from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe_1d import *
from methodsnm.fe_2d import *
from methodsnm.fe_4d import *

class FESpace(ABC):
    """
    Abstract base class for finite element spaces.
    """
    ndof = None
    mesh = None
    def __init__(self, mesh):
        pass

    @abstractmethod
    def _finite_element(self, elnr):
        raise Exception("Not implemented - Base class should not be used")

    @abstractmethod
    def _element_dofs(self, elnr):
        raise Exception("Not implemented - Base class should not be used")

    def _bndry_finite_element(self, elnr):
        raise Exception("_bndry_finite_element not implemented - Base class should not be used")

    def _bndry_element_dofs(self, elnr):
        raise Exception("_bndry_element_dofs not implemented - Base class should not be used")

    def finite_element(self, elnr, bndry=False):
        """
        Returns the finite element for the given element number.
        """
        if bndry:
            return self._bndry_finite_element(elnr)
        else:
            return self._finite_element(elnr)

    def element_dofs(self, elnr, bndry=False):
        """
        Returns the dofs for the given element number.
        """
        if bndry:
            return self._bndry_element_dofs(elnr)
        else:
            return self._element_dofs(elnr)

class VertexFirstSpace_1D(FESpace):
    def __init__(self, mesh):
        self.sfe = Node_FE()
        self.mesh = mesh

    def _bndry_element_dofs(self, bndry_elnr):
        return self.mesh.elements(bndry=True)[bndry_elnr]

    def _bndry_finite_element(self, bndry_elnr):
        return self.sfe

class P1_Segments_Space(VertexFirstSpace_1D):

    def __init__(self, mesh, periodic=False):
        super().__init__(mesh)
        self.periodic = periodic
        if periodic:
            self.ndof = len(mesh.points) - 1 
        else:
            self.ndof = len(mesh.points)
        self.fe = P1_Segment_FE()

    def _finite_element(self, elnr):
        return self.fe

    def _element_dofs(self, elnr):
        dofs = self.mesh.edges[elnr]
        if self.periodic and elnr == len(self.mesh.edges) - 1:
            return [dofs[0],0]
        else:
            return dofs

    def _bndry_element_dofs(self, bndry_elnr):
        dofs = self.mesh.elements(bndry=True)[bndry_elnr]
        if self.periodic:
            return [0]
        else:
            return dofs

class P1disc_Segments_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = 2*len(mesh.edges)
        self.mesh = mesh
        self.fe = P1_Segment_FE()

    def _finite_element(self, elnr):
        return self.fe

    def _element_dofs(self, elnr):
        return [2*elnr, 2*elnr+1]

    def _bndry_finite_element(self, bndry_elnr):
        raise Exception("No boundary evaluation for Discontinuous elements")        

    def _bndry_element_dofs(self, bndry_elnr):
        raise Exception("No boundary evaluation for Discontinuous elements")        

class Lagrange_Segments_Space(VertexFirstSpace_1D):

    def __init__(self, mesh, order=1):
        super().__init__(mesh)
        self.nv = len(mesh.points)
        self.ne = len(mesh.edges)
        self.order = order
        self.ndof = self.nv + self.ne*(order-1)
        self.fe=Lagrange_Segment_FE(order=self.order)

    def _finite_element(self, elnr):
        return self.fe

    def _element_dofs(self, elnr):
        offset = self.nv + elnr*(self.order-1)
        dofs = [self.mesh.edges[elnr][0]] + [offset + i for i in range(0,self.order-1)] + [self.mesh.edges[elnr][1]]
        return dofs



class Pk_IntLeg_Segments_Space(VertexFirstSpace_1D):

    def __init__(self, mesh, order=1):
        super().__init__(mesh)
        self.nv = len(mesh.points)
        self.ne = len(mesh.edges)
        self.order = order
        self.ndof = self.nv + self.ne*(order-1)
        self.fe=IntegratedLegendre_Segment_FE(order=self.order)

    def _finite_element(self, elnr):
        return self.fe

    def _element_dofs(self, elnr):
        offset = self.nv + elnr*(self.order-1)
        dofs = [self.mesh.edges[elnr][0]] + [self.mesh.edges[elnr][1]] + [offset + i for i in range(0,self.order-1)]
        return dofs



class P1_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points)
        self.mesh = mesh
        self.fe = P1_Triangle_FE()
        self.sfe = P1_Segment_FE()

    def _finite_element(self, elnr):
        return self.fe

    def _bndry_finite_element(self, bndry_elnr):
        return self.sfe

    def _element_dofs(self, elnr):
        return self.mesh.elements()[elnr]

    def _bndry_element_dofs(self, bndry_elnr):
        return self.mesh.elements(bndry=True)[bndry_elnr]


class P2_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points) + len(mesh.edges)
        self.nv = len(mesh.points)
        self.mesh = mesh
        self.fe = P2_Triangle_FE()
        self.sfe = Lagrange_Segment_FE(order=2)

    def _finite_element(self, elnr):
        return self.fe

    def _bndry_finite_element(self, bndry_elnr):
        return self.sfe

    def _element_dofs(self, elnr):
        return np.append(self.mesh.elements()[elnr], [self.nv + i for i in self.mesh.faces2edges[elnr]])

    def _bndry_element_dofs(self, bndry_elnr):
        verts = self.mesh.elements(bndry=True)[bndry_elnr]
        edge = self.mesh.bndry_edges[bndry_elnr]
        return [verts[0], self.nv + edge, verts[1]]

class P3_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points) + 2 * len(mesh.edges) + len(mesh.faces)
        self.nv = len(mesh.points)
        self.ned = len(mesh.edges)
        self.nf = len(mesh.faces)
        self.mesh = mesh
        self.fe = P3_Triangle_FE()
        self.sfe = Lagrange_Segment_FE(order=3)

    def _finite_element(self, elnr):
        return self.fe

    def _element_dofs(self, elnr):
        dofs = np.empty(10, dtype=int)
        vnums = self.mesh.elements()[elnr]
        dofs[0:3] = vnums
        j = 3
        enums = self.mesh.faces2edges[elnr]
        ref_verts_list = [[1,2],[0,2],[0,1]]
        for ref_edge, edge in enumerate(enums):
            tverts = vnums[ref_verts_list[ref_edge]]
            if (tverts[0] < tverts[1]):
                dofs[j] = self.nv + 2*edge + 0
                j += 1
                dofs[j] = self.nv + 2*edge + 1
                j += 1
            else:
                dofs[j] = self.nv + 2*edge + 1
                j += 1
                dofs[j] = self.nv + 2*edge + 0
                j += 1
        dofs[9] = self.nv + 2 * self.ned + elnr
        return dofs

    def _bndry_finite_element(self, bndry_elnr):
        return self.sfe

    def _bndry_element_dofs(self, bndry_elnr):
        verts = self.mesh.elements(bndry=True)[bndry_elnr]
        edge = self.mesh.bndry_edges[bndry_elnr]
        return [verts[0], self.nv + 2 * edge, self.nv + 2 * edge + 1, verts[1]]


class P1Edge_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.edges)
        self.mesh = mesh
        self.fe = P1Edge_Triangle_FE()

    def _finite_element(self, elnr):
        return self.fe

    def _element_dofs(self, elnr):
        edges = self.mesh.faces2edges[elnr]
        return edges
    
class P1_Tesserakt_Space(FESpace):
    """
    This class represents a P1 Tesserakt finite element space.
    """
    def __init__(self, mesh):
        self.ndof = len(mesh.points)
        self.mesh = mesh
        self.fe = P1_Tesserakt_FE()
        

    def _finite_element(self,elnr):
        return self.fe

    def _element_dofs(self, elnr):
        return self.mesh.elements()[elnr]


