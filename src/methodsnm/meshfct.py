from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.trafo import *
from methodsnm.vectorspace import *

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
        ip = np.asarray(ip)
        a  = trafo.points[0]   
        A  = trafo.jac           
        xi = np.linalg.solve(A, ip - a)   
        ip_ref = xi                       
        vals = fe.evaluate(ip_ref)
        return np.dot(vals, self.vector[dofs])

    def _evaluate_array1(self, ips, trafo=None):
        ips = np.asarray(ips)

        if trafo is not None:
            npts = ips.shape[0]
            for i in range(npts):
                ip = ips[i]
                local_trafo = trafo

            fe   = self.fes.finite_element(local_trafo.elnr)
            dofs = self.fes.element_dofs(local_trafo.elnr)

            return np.dot(fe.evaluate(ips), self.vector[dofs])

        vals = np.empty(len(ips))
        for i, ip in enumerate(ips):
            vals[i] = self._evaluate(ip, None)
        return vals
    
    def _evaluate_array(self, ips, trafo=None):
        ips = np.asarray(ips)
        npts = ips.shape[0]
        for i in range(npts):
            ip = ips[i]
            if trafo is None:
                el = self.fes.mesh.find_element(ip)
                local_trafo = self.fes.mesh.trafo(el)
            else:
                local_trafo = trafo
        fe = self.fes.finite_element(local_trafo.elnr)
        dofs = self.fes.element_dofs(local_trafo.elnr)
        return np.dot(fe.evaluate(ips), self.vector[dofs])




    def _set(self, f , boundary=False,bndry=None):
        """
        Sets the values of the finite element function to f.
        """
        if str(self.fes.fe) not in ["P1 Triangle Finite Element","P1 Tesserakt Finite Element","P1 4D-Simplex Finite Element"]:
            raise Exception("Only P1 triangle finite element,P1 Tesserakt Finite Element,P1 4D-Simplex Finite Element is supported")
        if boundary:
            for dof in self.mesh.bndry_vertices:
                x = self.mesh.points[dof]
                self.vector[dof] = f(x)
        elif bndry is None:
            for dof in self.mesh.vertices:
                x = self.mesh.points[dof]
                self.vector[dof] = f(x)
        elif bndry is not None:
            for dof in bndry:
                x = self.mesh.points[dof]
                self.vector[dof] = f(x)  

    def _set_P2(self, f, boundary=False, bndry=None):
        """
        Set values for a P2 Hypertriangle FE function.
        - Vertex-DOFs: f(x_v)
        - Edge-DOFs:   f(midpoint(v1,v2))
        """

        mesh = self.mesh
        nv = len(mesh.points)      # number of vertex DOFs
        ne = len(mesh.edges)       # number of edge DOFs

        if boundary:
            bset = set(mesh.bndry_vertices)

            for v in mesh.bndry_vertices:
                x = mesh.points[v]
                self.vector[v] = f(x)

            for eid, (v1, v2) in enumerate(mesh.edges):
                if v1 in bset and v2 in bset:
                    midpoint = 0.5*(mesh.points[v1] + mesh.points[v2])
                    self.vector[nv + eid] = f(midpoint)
            return

        if bndry is not None:
            bset = set(bndry)

            for v in bndry:
                if v < nv:   
                    x = mesh.points[v]
                    self.vector[v] = f(x)

            for dof in bndry:
                if dof >= nv:
                    eid = dof - nv
                    v1, v2 = mesh.edges[eid]
                    midpoint = 0.5*(mesh.points[v1] + mesh.points[v2])
                    self.vector[dof] = f(midpoint)
            return

        for v in range(nv):
            x = mesh.points[v]
            self.vector[v] = f(x)


        for eid, (v1, v2) in enumerate(mesh.edges):
            midpoint = 0.5*(mesh.points[v1] + mesh.points[v2])
            self.vector[nv + eid] = f(midpoint)
 

class VectorFunction(MeshFunction):
    """
    Abstract base class for vector-valued functions defined on a mesh.
    """
    def _evaluate(self, ip, trafo):
        raise Exception("Not implemented - Base class for vector functions")

    def _evaluate_array(self, ips, trafo):
        ret = np.empty((ips.shape[0], *self.value_shape))
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

class ConstantVectorFunction(VectorFunction):
    def __init__(self, vec, mesh=None):
        self.mesh = mesh
        self.vec = np.array(vec)
        self.value_shape = self.vec.shape  # wichtig für evaluate_array

    def _evaluate(self, ip, trafo):
        return self.vec

class FEVectorFunction(FEFunction):
    fes = None
    vector = None
    def __init__(self, fes):
        if not isinstance(fes, Productspace):
            raise TypeError("FEVectorFunction requires a Productspace FESpace.")
        self.mesh = fes.mesh
        self.fes = fes
        self.vector = np.zeros(fes.ndof)

    def block(self, b):
        return self.vector[self.fes.block_range(b)]
    
    def blocks(self):
        comps = []
        offset = 0
        for V in self.fes.spaces:
            n = V.ndof
            f = FEFunction(V)
            f.vector = self.vector[offset : offset+n].copy()
            comps.append(f)
            offset += n
        return comps

    def _evaluate(self, ip, trafo=None):
        """
        Gibt einen Vektor zurück: [u1(ip), u2(ip), ..., uk(ip)]
        """
        vals = []
        for b, V in enumerate(self.fes.spaces):
            u_loc = FEFunction(V)
            u_loc.vector = self.vector[self.fes.block_range(b)]
            if trafo is None:
                el = self.mesh.find_element(ip)
                trafo = self.mesh.trafo(el)
            vals.append(u_loc._evaluate(ip, trafo))
        return np.array(vals)
    
    def _evaluate_array(self, ips, trafo=None):
        ips = np.asarray(ips)
        npts = ips.shape[0]
        ncomp = len(self.fes.spaces)
        out = np.zeros((npts, ncomp))

        for i in range(npts):
            ip = ips[i]
            if trafo is None:
                el = self.fes.mesh.find_element(ip)
                local_trafo = self.fes.mesh.trafo(el)
            else:
                local_trafo = trafo

            for b, V in enumerate(self.fes.spaces):
                u_loc = FEFunction(V)

                u_loc.vector = self.vector[self.fes.block_range(b)]

                out[i, b] = u_loc._evaluate(ip, local_trafo)

        return out
    def _set(self, compdict):
        """
        Sets the values of the finite element function.
        compdict: dict
        Keys = component indices (int)
        Values = scalar functions (callables)
        """
        if not isinstance(self.fes, Productspace):
            raise TypeError("_set_vector can only be used with ProductSpace.")
        
        for b, fb in compdict.items():
                if b < 0 or b >= len(self.fes.spaces):
                    raise ValueError(f"Component index {b} out of range.")
                
                V = self.fes.spaces[b] #local Space

                u_loc = FEFunction(V) # local FE-Funktion
                if isinstance(fb,tuple):
                    func, bndry = fb
                    u_loc._set(func, bndry=bndry)
                else:
                    u_loc._set(fb)

                self.vector[self.fes.block_range(b)] = u_loc.vector
    
    def _set_P2(self, compdict):
        """
        Sets the values of the finite element function.
        compdict: dict
        Keys = component indices (int)
        Values = scalar functions (callables)
        """
        if not isinstance(self.fes, Productspace):
            raise TypeError("_set_vector can only be used with ProductSpace.")
        
        for b, fb in compdict.items():
                if b < 0 or b >= len(self.fes.spaces):
                    raise ValueError(f"Component index {b} out of range.")
                
                V = self.fes.spaces[b] #local Space

                u_loc = FEFunction(V) # local FE-Funktion
                if isinstance(fb,tuple):
                    func, bndry = fb
                    u_loc._set_P2(func, bndry=bndry)
                else:
                    u_loc._set_P2(fb)

                self.vector[self.fes.block_range(b)] = u_loc.vector