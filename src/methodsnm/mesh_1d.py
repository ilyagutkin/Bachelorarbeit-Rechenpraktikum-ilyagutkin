from abc import ABC, abstractmethod
import numpy as np
from numpy import array

from methodsnm.mesh import Mesh

from methodsnm.trafo import IntervalTransformation

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
