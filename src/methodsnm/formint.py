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
        dshapes_ref_test[:,-1,:] = 0
        dshapes_ref_trial = fe_trial.evaluate(intrule.nodes, deriv=True)
        dshapes_ref_trial[:,-1,:] = 0
        F = trafo.jacobian(intrule.nodes)
        invF = array([inv(F[i,:,:]) for i in range(F.shape[0])])
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)
        ret = einsum("ijk,imn,ijl,iml,i,i,i->kn", dshapes_ref_test, dshapes_ref_trial, invF, invF, adetF, coeffs, intrule.weights)
        return ret
    
class TimeIntegral(BilinearFormIntegral):
    def __init__(self, coeff = ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule = None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)
        gradu = fe_test.evaluate(intrule.nodes, deriv=True)
        ut = gradu[:, -1, :]
        v = fe_trial.evaluate(intrule.nodes)
        F = trafo.jacobian(intrule.nodes)
        invF = array([inv(F[i,:,:]) for i in range(F.shape[0])])
        invF_last = invF[:, -1, -1]
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)
        ret = einsum("ij,ik,i,i,i,i->jk",  v, ut, invF_last, adetF, coeffs, intrule.weights)
        return ret
    