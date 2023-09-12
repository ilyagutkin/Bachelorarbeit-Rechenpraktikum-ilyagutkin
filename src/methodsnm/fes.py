from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe_1d import *

class FESpace:
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

    def __init__(self, mesh):
        self.ndof = len(mesh.vertices)
        self.mesh = mesh

    def finite_element(self, elnr):
        return P1_Segment_FE()

    def element_dofs(self, elnr):
        return self.mesh.edges[elnr]

