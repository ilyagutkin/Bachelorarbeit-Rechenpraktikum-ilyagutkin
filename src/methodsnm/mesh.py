from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.trafo import *

class Mesh(ABC):
    dimension = None
    vertices = None
    edges = None
    faces = None

    def __init__(self):
        raise NotImplementedError("Not implemented")

    def elements(self):
        if self.dimension == 1:
            return self.edges
        elif self.dimension == 2:   
            return self.faces
        else:
            raise Exception("Invalid dimension")

    def trafo(self, elnr):
        raise NotImplementedError("Not implemented")

class Mesh1D(Mesh):
    def __init__(self, vertices, sub_intervals=1):
        self.dimension = 1
        ne = (len(vertices)-1) * sub_intervals
        nv = ne + 1
        self.vertices = np.empty(nv)
        nme = len(vertices)-1
        for i in range(nme):
            for j in range(sub_intervals):
                self.vertices[i*sub_intervals+j] = vertices[i] + j*(vertices[i+1]-vertices[i])/sub_intervals
        self.vertices[-1] = vertices[-1]
        self.edges = np.array([[i,i+1] for i in range(ne)])

    def trafo(self, elnr):
        return IntervalTransformation(self, elnr)

    def uniform_refine(self):
        return Mesh1D(self.vertices, sub_intervals=2)
