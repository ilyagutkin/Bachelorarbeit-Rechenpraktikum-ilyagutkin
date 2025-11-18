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
        
    def boundary_vertices(self):
        return set(self.mesh.bndry_vertices)

    def initial_vertices(self):
        return set(self.mesh.initial_bndry_vertices)

    def top_vertices(self):
        return set(self.mesh.top_bndry_vertices)
    
    def boundary_dofs_excluding_top(self):
        v = self.boundary_vertices() - self.top_vertices()
        return sorted(self.boundary_dofs_from_vertex_set(v))

    def boundary_dofs_from_vertex_set(self, vset):
        return sorted(vset)

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

class P1_Hypertriangle_Space(FESpace):
    """
    This class represents a P1 Hypertriangle finite element space.
    """
    def __init__(self, mesh):
        self.ndof = len(mesh.points)
        self.mesh = mesh
        self.fe = P1_Hypertriangle_FE()

    def _finite_element(self,elnr):
        return self.fe

    def _element_dofs(self, elnr):
        return self.mesh.elements()[elnr]
    
class P2_Hypertriangle_Space(FESpace):
    """
    P2 space on 4D simplex mesh.
    DOFs: 1 per vertex + 1 per edge of each hyper-tetrahedron.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._build_edges_from_hypercells()
        self.nv = len(mesh.points)
        self.ne = len(self.mesh.edges)
        self.ndof = self.nv + self.ne
        self.fe = P2_Hypertriangle_FE()

    def _build_edges_from_hypercells(self):
        """
        Build global edge list and hypercells->edges connectivity
        from hypercells (each with 5 vertices).
        """
        # if edges already exist, reuse them
        if getattr(self.mesh, "edges", None) is not None and \
           getattr(self.mesh, "hypercells2edges", None) is not None:
            return

        cells = self.mesh.elements()   # hypercells, shape (ncells, 5)
        edge_dict = {}                 # (v1,v2) -> edge index
        cells2edges = []

        for el in cells:
            verts = list(el)
            local_edges = []
            # all pairs of 5 vertices -> 10 edges
            for a in range(5):
                for b in range(a+1, 5):
                    v1, v2 = verts[a], verts[b]
                    key = (v1, v2) if v1 < v2 else (v2, v1)
                    if key not in edge_dict:
                        edge_dict[key] = len(edge_dict)
                    local_edges.append(edge_dict[key])
            cells2edges.append(local_edges)

        # store on mesh for reuse
        self.mesh.edges = np.array(list(edge_dict.keys()), dtype=int)
        self.mesh.hypercells2edges = np.array(cells2edges, dtype=int)

    def _finite_element(self, elnr):
        return self.fe

    def _element_dofs(self, elnr):
        """
        Local DOF numbering:
        - first 5: vertex dofs (like P1)
        - last 10: edge dofs (global index shifted by nv)
        """
        verts = self.mesh.elements()[elnr]
        edge_ids = self.mesh.hypercells2edges[elnr]
        return np.concatenate([verts, self.nv + edge_ids])
    
    def boundary_dofs(self):
        v = self.boundary_vertices()
        return sorted(
            list(v) + self._edge_boundary_dofs(v)
        )
    
    def boundary_dofs_from_vertex_set(self, vset):
        v = sorted(vset)
        e = self._edge_boundary_dofs(vset)
        return sorted(v + e)

    def _edge_boundary_dofs(self, vertex_set):
        nv = self.nv
        edge_dofs = []
        for eid, (v1, v2) in enumerate(self.mesh.edges):
            if v1 in vertex_set and v2 in vertex_set:
                edge_dofs.append(nv + eid)
        return edge_dofs

