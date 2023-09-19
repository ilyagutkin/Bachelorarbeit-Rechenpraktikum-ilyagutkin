from abc import ABC, abstractmethod
import numpy as np
from numpy import array

from methodsnm.trafo import *

class Mesh(ABC):
    dimension = None
    points = None
    vertices = None
    edges = None
    faces = None

    bndry_vertices = None
    bndry_edges = None
    
    def __init__(self):
        raise NotImplementedError("Not implemented")

    def elements(self, codim=0, bndry=False):
        if self.dimension - codim == 0:
            if bndry:
                return self.vertices[self.bndry_vertices]
            else:
                return self.vertices
        elif self.dimension - codim == 1:   
            if bndry:
                return self.edges[self.bndry_edges]
            else:
                return self.edges
        elif self.dimension - codim == 2:   
            return self.faces
        else:
            raise Exception("Invalid dimension")

    def trafo(self, elnr, codim=0, bndry=False):
        raise NotImplementedError("Not implemented")

class Mesh1D(Mesh):
    def __init__(self, points, sub_intervals=1):
        self.dimension = 1
        ne = (len(points)-1) * sub_intervals
        nv = ne + 1
        self.points = np.empty(nv)
        nme = len(points)-1
        for i in range(nme):
            for j in range(sub_intervals):
                self.points[i*sub_intervals+j] = points[i] + j*(points[i+1]-points[i])/sub_intervals
        self.points[-1] = points[-1]
        self.vertices = np.arange(nv)
        self.edges = np.array([[i,i+1] for i in range(ne)])
        self.bndry_vertices = np.array([0,nv-1])
        self.bndry_edges = None

    def trafo(self, elnr, codim=0, bndry=False):
        if codim > 0 or bndry:
            raise NotImplementedError("Not implemented")
        return IntervalTransformation(self, elnr)

    def uniform_refine(self):
        return Mesh1D(self.points, sub_intervals=2)

class Mesh2D(Mesh):
    def __init__(self):
        self.dimension = 2

class StructuredRectangleMesh(Mesh2D):
    def __init__(self, M, N, mapping = None):
        super().__init__()
        if mapping is None:
            mapping = lambda x,y: [x,y]
        self.points = np.array([array(mapping(i/M,j/N)) for j in range(N+1) for i in range(M+1)])
        self.vertices = np.arange((M+1)*(N+1))
        self.faces = np.array([[    j*(M+1)+i,     j*(M+1)+i+1, (j+1)*(M+1)+i] for i in range(M) for j in range(N)] + 
                              [[  j*(M+1)+i+1, (j+1)*(M+1)+i+1, (j+1)*(M+1)+i] for i in range(M) for j in range(N)])
        self.edges = np.array([[    j*(M+1)+i,     j*(M+1)+i+1] for j in range(N+1) for i in range(M) ] + 
                              [[    j*(M+1)+i,   (j+1)*(M+1)+i] for i in range(M+1) for j in range(N)] + 
                              [[(j+1)*(M+1)+i,     j*(M+1)+i+1] for i in range(M) for j in range(N)])
        self.bndry_vertices = [i for i in range(M)] + [M+j*(M+1) for j in range(N)] \
                              + [(N+1)*(M+1)-i-1 for i in range(M)] + [(N-j)*(M+1) for j in range(N)]
        self.bndry_edges = [i for i in range(M)] + [2*N*M+M+j for j in range(N)] \
                              + [N*M+M-1-i for i in range(M)] + [(N+1)*M+N-1-j for j in range(N)]
