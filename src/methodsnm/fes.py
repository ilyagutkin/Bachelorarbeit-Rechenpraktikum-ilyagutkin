from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.fe_1d import *

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

try: 
    from methodsnm.solution.fes_sol import *
except:
    pass