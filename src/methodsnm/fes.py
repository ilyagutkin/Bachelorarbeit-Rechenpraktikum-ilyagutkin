from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe_1d import *
from methodsnm.fe_2d import *

class FESpace:
    """
    Abstract base class for finite element spaces.
    """
    ndof = None
    mesh = None
    def __init__(self, mesh):
        pass

    @abstractmethod
    def finite_element(self, elnr):
        raise Exception("Not implemented - Base class should not be used")

    @abstractmethod
    def element_dofs(self, elnr):
        raise Exception("Not implemented - Base class should not be used")


class P1_Segments_Space(FESpace):

    def __init__(self, mesh, periodic=False):
        self.periodic = periodic
        if periodic:
            self.ndof = len(mesh.points) - 1 
        else:
            self.ndof = len(mesh.points)
        self.mesh = mesh
        self.fe = P1_Segment_FE()

    def finite_element(self, elnr):
        return self.fe

    def element_dofs(self, elnr):
        dofs = self.mesh.edges[elnr]
        if self.periodic and elnr == len(self.mesh.edges) - 1:
            return [dofs[0],0]
        else:
            return dofs

class P1_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points)
        self.mesh = mesh
        self.fe = P1_Triangle_FE()

    def finite_element(self, elnr):
        return self.fe

    def element_dofs(self, elnr):
        return self.mesh.elements()[elnr]

class P1disc_Segments_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = 2*len(mesh.edges)
        self.mesh = mesh
        self.fe = P1_Segment_FE()

    def finite_element(self, elnr):
        return self.fe

    def element_dofs(self, elnr):
        return [2*elnr, 2*elnr+1]

class Lagrange_Segments_Space(FESpace):

    def __init__(self, mesh, order=1):
        self.nv = len(mesh.points)
        self.ne = len(mesh.edges)
        self.order = order
        self.ndof = self.nv + self.ne*(order-1)
        self.mesh = mesh
        self.fe=Lagrange_Segment_FE(order=self.order)

    def finite_element(self, elnr):
        return self.fe

    def element_dofs(self, elnr):
        offset = self.nv + elnr*(self.order-1)
        dofs = [self.mesh.edges[elnr][0]] + [offset + i for i in range(0,self.order-1)] + [self.mesh.edges[elnr][1]]
        return dofs


class Pk_IntLeg_Segments_Space(FESpace):

    def __init__(self, mesh, order=1):
        self.nv = len(mesh.points)
        self.ne = len(mesh.edges)
        self.order = order
        self.ndof = self.nv + self.ne*(order-1)
        self.mesh = mesh
        self.fe=IntegratedLegendre_Segment_FE(order=self.order)

    def finite_element(self, elnr):
        return self.fe

    def element_dofs(self, elnr):
        offset = self.nv + elnr*(self.order-1)
        dofs = [self.mesh.edges[elnr][0]] + [self.mesh.edges[elnr][1]] + [offset + i for i in range(0,self.order-1)]
        return dofs

class P1_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points)
        self.mesh = mesh
        self.fe = P1_Triangle_FE()

    def finite_element(self, elnr):
        return self.fe

    def element_dofs(self, elnr):
        return self.mesh.elements()[elnr]


class P2_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points) + len(mesh.edges)
        self.nv = len(mesh.points)
        self.mesh = mesh
        self.fe = P2_Triangle_FE()

    def finite_element(self, elnr):
        return self.fe

    def element_dofs(self, elnr):
        return np.append(self.mesh.elements()[elnr], [self.nv + i for i in self.mesh.faces2edges[elnr]])
      
class P3_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.points) + 2 * len(mesh.edges) + len(mesh.faces)
        self.nv = len(mesh.points)
        self.ned = len(mesh.edges)
        self.nf = len(mesh.faces)
        self.mesh = mesh
        self.fe = P3_Triangle_FE()

    def finite_element(self, elnr):
        return self.fe

    def element_dofs(self, elnr):
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
        
class P1Edge_Triangle_Space(FESpace):

    def __init__(self, mesh):
        self.ndof = len(mesh.edges)
        self.mesh = mesh
        self.fe = P1Edge_Triangle_FE()

    def finite_element(self, elnr):
        return self.fe

    def element_dofs(self, elnr):
        edges = self.mesh.faces2edges[elnr]
        return edges
