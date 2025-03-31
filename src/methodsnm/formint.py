from abc import ABC, abstractmethod
import numpy as np
from numpy import array, einsum
from methodsnm.intrule import *
from methodsnm.fe import *
from numpy.linalg import det, inv
from methodsnm.meshfct import ConstantFunction

class FormIntegral(ABC):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Not implemented")

class LinearFormIntegral(FormIntegral):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Not implemented")

    def compute_element_vector(self, fe, trafo):
        raise NotImplementedError("Not implemented")

class SourceIntegral(LinearFormIntegral):

    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_vector(self, fe, trafo, intrule = None):
        if intrule is None:
            intrule = select_integration_rule(2*fe.order, fe.eltype)
        shapes = fe.evaluate(intrule.nodes)
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)
        weights = [w*c*abs(det(trafo.jacobian(ip))) for ip,w,c in zip(intrule.nodes,intrule.weights,coeffs)]
        return np.dot(shapes.T, weights)

class BilinearFormIntegral(FormIntegral):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Not implemented")

    def compute_element_matrix(self, fe, trafo):
        raise NotImplementedError("Not implemented")

class MassIntegral(BilinearFormIntegral):

    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule = None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)
        shapes_test = fe_test.evaluate(intrule.nodes)
        shapes_trial = fe_test.evaluate(intrule.nodes)
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)
        F = trafo.jacobian(intrule.nodes)
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        ret = einsum("ij,ik,i,i,i->jk", shapes_test, shapes_trial, adetF, coeffs, intrule.weights)
        return ret

class LaplaceIntegral(BilinearFormIntegral):

    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule = None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)
        dshapes_ref_test = fe_test.evaluate(intrule.nodes, deriv=True)
        dshapes_ref_trial = fe_trial.evaluate(intrule.nodes, deriv=True)
        F = trafo.jacobian(intrule.nodes)
        invF = array([inv(F[i,:,:]) for i in range(F.shape[0])])
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)
        ret = einsum("ijk,imn,ijl,iml,i,i,i->kn", dshapes_ref_test, dshapes_ref_trial, invF, invF, adetF, coeffs, intrule.weights)
        return ret
    
class LaplaceIntegral2(BilinearFormIntegral):

    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule=None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            order = fe_test.order + fe_trial.order
            intrule = select_integration_rule(order, fe_test.eltype)

        # Ableitungen der Basisfunktionen im Referenzraum
        dshapes_ref_test = fe_test.evaluate(intrule.nodes, deriv=True)   # shape: (npoints, ndofs_test, dim)
        dshapes_ref_trial = fe_trial.evaluate(intrule.nodes, deriv=True) # shape: (npoints, ndofs_trial, dim)

        dshapes_ref_test = dshapes_ref_test[:, :, :2]
        dshapes_ref_trial = dshapes_ref_trial[:, :, :2]
        # Geometrische Transformation
        F = trafo.jacobian(intrule.nodes)         # shape: (npoints, dim, dim)
        invF = np.linalg.inv(F)                   # shape: (npoints, dim, dim)
        detF = np.abs(np.linalg.det(F))           # shape: (npoints,)
        
        # Transformation der Gradienten in Weltkoordinaten
        grad_test = einsum("pki,pij->pkj", dshapes_ref_test, invF)
        grad_trial = einsum("pni,pij->pnj", dshapes_ref_trial, invF)

        # Koeffizientenfunktion (z. B. konstantes 1)
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)  # shape: (npoints,)

        # Skalarprodukt der Gradienten
        scalar_prods = einsum("pkj,pnj->pkn", grad_test, grad_trial)

        # Gewichtung mit |det(J)|, Gewichte und Koeffizienten
        weights = intrule.weights                           # shape: (npoints,)
        factors = coeffs * detF * weights                   # shape: (npoints,)

        # Endliche Summation über Integrationspunkte
        elmat = einsum("pkn,p->kn", scalar_prods, factors)  # shape: (ndofs_test, ndofs_trial)

        return elmat
