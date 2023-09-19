from abc import ABC, abstractmethod
import numpy as np
from numpy import array

from methodsnm.mesh import Mesh
from methodsnm.trafo import TriangleTransformation

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

    def trafo(self, elnr, codim=0, bndry=False):
        if codim > 0 or bndry:
            raise NotImplementedError("Not implemented yet")
        return TriangleTransformation(self, elnr)
