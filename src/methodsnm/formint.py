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
    
class LaplaceIntegral_without_time(BilinearFormIntegral):

    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule = None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)
        dshapes_ref_test = fe_test.evaluate(intrule.nodes, deriv=True)
        #dshapes_ref_test[:,-1,:] = 0
        dshapes_ref_trial = fe_trial.evaluate(intrule.nodes, deriv=True)
        #dshapes_ref_trial[:,-1,:] = 0

        F = trafo.jacobian(intrule.nodes)
        invF = array([inv(F[i,:,:]) for i in range(F.shape[0])])
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)

    # Entferne die letzte Komponente entlang der Raumrichtung (axis=1)
        dshapes_ref_test = np.delete(dshapes_ref_test, -1, axis=1)     # [n_qp, dim-1, n_test]
        dshapes_ref_trial = np.delete(dshapes_ref_trial, -1, axis=1)   # [n_qp, dim-1, n_trial]
        invF = np.delete(invF, -1, axis=1)  

        ret = einsum("ijk,imn,ijl,iml,i,i,i->kn", dshapes_ref_test, dshapes_ref_trial, invF, invF, adetF, coeffs, intrule.weights)
        return ret

class LaplaceIntegral_without_time1(BilinearFormIntegral):
    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule=None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)

        dshapes_ref_test = fe_test.evaluate(intrule.nodes, deriv=True)    # [n_qp, dim, n_test]
        dshapes_ref_trial = fe_trial.evaluate(intrule.nodes, deriv=True)  # [n_qp, dim, n_trial]

        # Entferne Zeitableitung (letzte Raumrichtung)
        dshapes_ref_test = np.delete(dshapes_ref_test, -1, axis=1)        # [n_qp, dim-1, n_test]
        dshapes_ref_trial = np.delete(dshapes_ref_trial, -1, axis=1)

        F = trafo.jacobian(intrule.nodes)
        invF = array([inv(F[i,:,:].T) for i in range(F.shape[0])])        # [n_qp, dim, dim]
        Jinv = np.delete(invF, -1, axis=1)                                # [n_qp, dim-1, dim]
        
        grad_test = np.einsum("nij,njk->nik", Jinv, dshapes_ref_test)     # [n_qp, dim-1, n_test]
        grad_trial = np.einsum("nij,njk->nik", Jinv, dshapes_ref_trial)   # [n_qp, dim-1, n_trial]

        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])    # [n_qp]
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)                # [n_qp]
        weights = intrule.weights                                         # [n_qp]

        ret = np.einsum("nij,nil,i,i,i->lj", grad_test, grad_trial, adetF, coeffs, weights)
        return ret


class TimeIntegral(BilinearFormIntegral):
    def __init__(self, coeff = ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule = None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)
        shapes_trial = fe_test.evaluate(intrule.nodes, deriv =True)
        shapes_test = fe_trial.evaluate(intrule.nodes)
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)
        F = trafo.jacobian(intrule.nodes)
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        invF = array([inv(F[i,:,:].T) for i in range(F.shape[0])])
        w = einsum("nij,njb->nib", invF, shapes_trial) 
        dt_trial = w[:, -1,:]  
        ret = einsum("ij,ik,i,i,i->kj", dt_trial, shapes_test, adetF, coeffs, intrule.weights)
        return ret


   

