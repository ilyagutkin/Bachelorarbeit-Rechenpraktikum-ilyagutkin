from abc import ABC, abstractmethod
import numpy as np
from numpy import array

class Mesh(ABC):
    dimension = None
    points = None
    vertices = None
    edges = None
    faces = None

    face2edges = None

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

