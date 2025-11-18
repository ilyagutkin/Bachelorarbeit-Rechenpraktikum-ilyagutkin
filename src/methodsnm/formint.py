from abc import ABC, abstractmethod
import numpy as np
from numpy import sqrt
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
    
class SUPGSourceIntegral(LinearFormIntegral):
    def __init__(self, f=ConstantFunction(np.ones(1)),wind=None):
        self.coeff = f
        self.wind = wind

    def compute_element_vector(self, fe, trafo, intrule=None):
        if intrule is None:
            intrule = select_integration_rule(2*fe.order, fe.eltype)
        if trafo.mesh.dimension == 2:
            M = np.array([[2/sqrt(3), 1/sqrt(3)], [1/sqrt(3), 2/sqrt(3)]])
        elif trafo.mesh.dimension == 4:
            M = 5 **-1/4*np.array([
    [2, 1, 1, 1],
    [1, 2, 1, 1],
    [1, 1, 2, 1],
    [1, 1, 1, 2]])
        h = trafo.mesh.special_meshsize
        shapes = fe.evaluate(intrule.nodes,deriv=True)
        F = trafo.jacobian(intrule.nodes)
        invF = array([inv(F[i,:,:].T) for i in range(F.shape[0])])
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        wind = self.wind.evaluate(intrule.nodes, trafo)
        f = self.coeff.evaluate(intrule.nodes, trafo)
        cinv = 1
        tmp = np.einsum("nij,jk->nik", invF, M)
        expr = np.einsum("nij,nkj->nik", tmp, invF)
        tau = np.einsum("ni,nij,nj->n", wind, expr, wind)
        gamma_T = 1 / np.sqrt(tau + (cinv / h)**2)

        w = einsum("nij,njb->nib", invF, shapes) 
        conv = np.einsum("nd,ndi->ni", wind, w)

        scale = gamma_T * adetF * intrule.weights
        return np.einsum("n,n,nj->j", scale , f , conv )

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
        Jinv = np.array([inv(F[i, :, :].T)for i in range(F.shape[0])])
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)
        grad_test  = np.einsum("nij,njk->nik", Jinv, dshapes_ref_test)   # [n_qp, dim, n_test]
        grad_trial = np.einsum("nij,njk->nik", Jinv, dshapes_ref_trial)  # [n_qp, dim, n_trial]

        #ret = einsum("ijk,imn,ijl,iml,i,i,i->kn", dshapes_ref_test, dshapes_ref_trial, invF, invF, adetF, coeffs, intrule.weights)
        ret = np.einsum("ijk,ijl,i,i,i->kl",grad_test, grad_trial,adetF, coeffs, intrule.weights)
        return ret
    
class LaplaceIntegral_without_time(BilinearFormIntegral):
    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule=None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)

        dshapes_ref_test = fe_test.evaluate(intrule.nodes, deriv=True)    # [n_qp, dim, n_test]
        dshapes_ref_trial = fe_trial.evaluate(intrule.nodes, deriv=True)  # [n_qp, dim, n_trial]

        F = trafo.jacobian(intrule.nodes)
        Jinv = array([inv(F[i,:,:].T) for i in range(F.shape[0])])        # [n_qp, dim, dim]                              # [n_qp, dim-1, dim]
        
        grad_test = np.einsum("nij,njk->nik", Jinv, dshapes_ref_test)     # [n_qp, dim, n_test]
        grad_trial = np.einsum("nij,njk->nik", Jinv, dshapes_ref_trial)   # [n_qp, dim, n_trial]

        grad_test = np.delete(grad_test, -1, axis=1)     # [n_qp, dim-1, n_test]
        grad_trial = np.delete(grad_trial, -1, axis=1)   # [n_qp, dim-1, n_trial]

        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])    # [n_qp]
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)                # [n_qp]
        weights = intrule.weights                                         # [n_qp]

        ret = np.einsum("ijk,ijl,i,i,i->kl", grad_test, grad_trial, adetF, coeffs, weights)
        return ret


class TimeIntegral(BilinearFormIntegral):
    def __init__(self, coeff = ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule = None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)
        shapes_trial = fe_trial.evaluate(intrule.nodes, deriv =True)
        shapes_test = fe_test.evaluate(intrule.nodes)
        coeffs = self.coeff.evaluate(intrule.nodes, trafo)
        F = trafo.jacobian(intrule.nodes)
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        invF = array([inv(F[i,:,:].T) for i in range(F.shape[0])])
        w = einsum("nij,njb->nib", invF, shapes_trial) 
        dt_trial = w[:, -1,:]  
        ret = einsum("ij,ik,i,i,i->kj", dt_trial, shapes_test, adetF, coeffs, intrule.weights)
        return ret

class ConvectionIntegral(BilinearFormIntegral):
    def __init__(self, coeff=ConstantFunction(np.ones(1))):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule=None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)
        shapes_test = fe_test.evaluate(intrule.nodes)
        shapes_trial = fe_trial.evaluate(intrule.nodes, deriv=True)
        F = trafo.jacobian(intrule.nodes)
        invF = array([inv(F[i,:,:].T) for i in range(F.shape[0])])
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        wind = self.coeff.evaluate(intrule.nodes, trafo)

        w = einsum("nij,njb->nib", invF, shapes_trial) 
        conv = np.einsum("nd,ndi->ni", wind, w)

        ret = einsum("ij,ik,i,i->kj", conv , shapes_test, adetF, intrule.weights)
        return ret
   
class SUPGIntegral(BilinearFormIntegral):
    def __init__(self, coeff=ConstantFunction(np.ones(1))):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule=None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order, fe_test.eltype)
        if trafo.mesh.dimension == 2:
            M = np.array([[2/sqrt(3), 1/sqrt(3)], [1/sqrt(3), 2/sqrt(3)]])
        elif trafo.mesh.dimension == 4:
            M = 5 **-1/4*np.array([
    [2, 1, 1, 1],
    [1, 2, 1, 1],
    [1, 1, 2, 1],
    [1, 1, 1, 2]])
        h = trafo.mesh.special_meshsize
        shapes_test = fe_test.evaluate(intrule.nodes, deriv=True)
        shapes_trial = fe_trial.evaluate(intrule.nodes, deriv=True)
        F = trafo.jacobian(intrule.nodes)
        invF = array([inv(F[i,:,:].T) for i in range(F.shape[0])])
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        wind = self.coeff.evaluate(intrule.nodes, trafo)
        cinv =1
        tmp = np.einsum("nij,jk->nik", invF, M)
        expr = np.einsum("nij,nkj->nik", tmp, invF)
        tau = np.einsum("ni,nij,nj->n", wind, expr, wind)
        gamma_T = 1 / np.sqrt(tau + (cinv / h)**2)

        grad_u_phys = np.einsum("nij,njb->nib", invF, shapes_trial)
        grad_v_phys = np.einsum("nij,njb->nib", invF, shapes_test)

        wind_grad_u = np.einsum("nd,ndi->ni", wind, grad_u_phys)
        wind_grad_v = np.einsum("nd,ndi->ni", wind, grad_v_phys)

        scale = gamma_T * adetF * intrule.weights  # (n,)
        ret = np.einsum("n,ni,nj->ij", scale, wind_grad_u, wind_grad_v)
        return ret

class DivUQIntegrator(BilinearFormIntegral):
    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule=None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have same type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order,fe_test.eltype)

        shapes_q = fe_test.evaluate(intrule.nodes)  # [n_qp, ndof_q]
        shapes_u_ref = fe_trial.evaluate(intrule.nodes, deriv=True)  # [n_qp, dim, ndof_u]
        F = trafo.jacobian(intrule.nodes)            # [n_qp, dim, dim]
        invF = array([inv(F[i,:,:].T) for i in range(F.shape[0])])
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        weights = intrule.weights
        grad_u = einsum("nij,njb->nib", invF, shapes_u_ref) 
        div_u = grad_u[:, :3, :].sum(axis=1)
        ret = einsum("ik,il,i,i->kl", shapes_q, div_u, adetF, weights)

        return ret

class DivVPIntegrator(BilinearFormIntegral):
    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule=None):
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have same type")
            intrule = select_integration_rule(fe_test.order + fe_trial.order,fe_test.eltype)

        shapes_v = fe_test.evaluate(intrule.nodes,deriv =True)  # [n_qp, ndof_v]
        shapes_p = fe_trial.evaluate(intrule.nodes)  # [n_qp, dim, ndof_q]
        F = trafo.jacobian(intrule.nodes)            # [n_qp, dim, dim]
        invF = array([inv(F[i,:,:].T) for i in range(F.shape[0])])
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        weights = intrule.weights
        grad_v = einsum("nij,njb->nib", invF, shapes_v)  # [n_qp, dim, ndof_q]
        div_v = grad_v[:, :3, :].sum(axis=1)  # [n_qp, ndof_q]
        ret = einsum("il,ik,i,i->lk",   div_v , shapes_p, adetF, weights)

        return ret

class PressureStabilizationIntegral(BilinearFormIntegral):
    """
    Simple pressure-gradient stabilization:
        s(p, q) = delta * (∇p, ∇q)
    """

    def __init__(self, delta = 1.0):
        self.delta = delta

    def compute_element_matrix(self, fe_test, fe_trial, trafo, intrule=None):
        # EXACT same structure as LaplaceIntegral_without_time, but with coeff = delta
        if intrule is None:
            if fe_test.eltype != fe_trial.eltype:
                raise Exception("Finite elements must have the same el. type")
            intrule = select_integration_rule(
                fe_test.order + fe_trial.order,
                fe_test.eltype
            )

        # Values / gradients of basis
        shapes_test = fe_test.evaluate(intrule.nodes, deriv=True)   # (n_qp, dim, ndof)
        shapes_trial = fe_trial.evaluate(intrule.nodes, deriv=True)

        # Geometry
        F = trafo.jacobian(intrule.nodes)
        invF = array([inv(F[i,:,:].T) for i in range(F.shape[0])])
        adetF = array([abs(det(F[i,:,:])) for i in range(F.shape[0])])
        weights = intrule.weights

        # map reference gradients to physical gradients
        grad_test  = einsum("nij,njb->nib", invF, shapes_test)       # (n_qp, dim, ndof_test)
        grad_trial = einsum("nij,njb->nib", invF, shapes_trial)  

        # contraction: ∫ grad_test · grad_trial * |detJ| * w
        ret = self.delta * einsum("ndk,ndl,n,n->kl",
                                   grad_test, grad_trial, adetF, weights)
        h = trafo.mesh.special_meshsize

        return -h**2*ret

