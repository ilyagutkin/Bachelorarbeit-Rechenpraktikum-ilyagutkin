from abc import ABC, abstractmethod
import numpy as np
from numpy import array
from methodsnm.trafo import *
from methodsnm.mesh import *
from scipy import sparse

class Form(ABC):
    integrals = None
    fes = None
    def __init__(self):
        raise NotImplementedError("Not implemented")

    def add_integrator(self, integrator):
        self.integrals.append(integrator)

    def __iadd__(self, integrator):
        self.add_integrator(integrator)
        return self

class LinearForm(Form):

    vector = None
    def __init__(self, fes=None):
        self.fes = fes
        self.integrals = []
    
    def assemble(self):
        #raise NotImplementedError("Not implemented")
        self.vector = np.zeros(self.fes.ndof)
        mesh = self.fes.mesh
        for elnr, verts in enumerate(mesh.elements()):
            trafo = mesh.trafo(elnr)
            fe = self.fes.finite_element(elnr)
            dofs = self.fes.element_dofs(elnr)
            for integral in self.integrals:
                self.vector[dofs] += integral.compute_element_vector(fe, trafo)


class BilinearForm(Form):

    matrix = None
    def __init__(self, fes=None, fes_test=None, fes_trial=None):
        if fes is None and (fes_test is None or fes_trial is None) or all([f is not None for f in [fes,fes_test,fes_trial]]):
            raise Exception("Invalid arguments, specify either `fes` or `fes_test` and `fes_trial`")
        if fes_test is None:
            fes_test = fes
            fes_trial = fes

        self.fes_trial = fes_trial
        self.fes_test = fes_test
        self.fes = fes_test
        self.integrals = []
    
    def assemble(self):
        #raise NotImplementedError("Not implemented")
        self.matrix = sparse.lil_matrix((self.fes_test.ndof, self.fes_trial.ndof))
        mesh = self.fes.mesh
        for elnr, verts in enumerate(mesh.elements()):
            trafo = mesh.trafo(elnr)
            fe_test = self.fes_test.finite_element(elnr)
            dofs_test = self.fes_test.element_dofs(elnr)
            fe_trial = self.fes_trial.finite_element(elnr)
            dofs_trial = self.fes_trial.element_dofs(elnr)
            elmat = np.zeros((len(dofs_test), len(dofs_trial)))
            for integral in self.integrals:
                elmat += integral.compute_element_matrix(fe_test, fe_trial, trafo)
            for i, dofi in enumerate(dofs_test):
                for j, dofj in enumerate(dofs_trial):
                    self.matrix[dofi,dofj] += elmat[i,j]
        self.matrix = self.matrix.tocsr()

