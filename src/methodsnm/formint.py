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
        """Compute the local element vector for a linear form.

        Contract: subclasses should implement this method and return the
        element load/vector b with components
            b_i = \int_K rhs(x) \; \phi_i(x) \; dx
        (or a stabilized variant). Implementations may accept an
        optional quadrature rule argument (``intrule``) and should
        follow the signature used in this module: the finite element
        descriptor ``fe`` and a transformation ``trafo`` are passed in
        and can be used to evaluate shapes, gradients, and to map
        quadrature points to the physical element.

        Parameters
        ----------
        fe : FiniteElement
            Finite element descriptor (shape functions, order, etc.).
        trafo : object
            Transformation object providing Jacobian mapping and mesh
            information.

        Returns
        -------
        ndarray
            Local element vector (length = number of local DOFs).
        """
        raise NotImplementedError("Not implemented")

class SourceIntegral(LinearFormIntegral):

    def __init__(self, coeff=ConstantFunction(1)):
        self.coeff = coeff

    def compute_element_vector(self, fe, trafo, intrule = None):
        """Compute the element load vector for a source term.

        Weak formulation (element K):
            b_i = \int_K f(x) \; \phi_i(x) \; dx

        where f(x) is the source coefficient (``self.coeff``) and
        \phi_i are the test basis functions associated with ``fe``.
        The method evaluates the shape functions at quadrature points,
        evaluates the coefficient at those points (using ``trafo`` to
        map to physical coordinates), multiplies by the Jacobian
        determinant and quadrature weights, and returns the assembled
        local vector.

        Parameters
        ----------
        fe : FiniteElement
            Finite element descriptor used for the test space.
        trafo : object
            Transformation object providing mapping and Jacobian.
        intrule : IntegrationRule, optional
            Quadrature rule; if None a rule is chosen based on the
            polynomial order.

        Returns
        -------
        ndarray
            Local element right-hand-side vector of shape (n_local_dofs,).
        """
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
        """Compute the SUPG-stabilized element load vector.

        This routine computes the streamline-upwind Petrov-Galerkin
        stabilization contribution to the right-hand side. The
        assembled vector corresponds to the stabilized integral
            b_i^{SUPG} = \int_K \gamma_T(x) \, f(x) \, (w(x)\cdot\nabla\phi_i(x)) \; dx

        where
          - f(x) is the source function (``self.coeff``),
          - w(x) is the convective velocity (``self.wind``), and
          - \gamma_T is a local stabilization parameter depending on
            the element metric and mesh size h.

        Implementation notes:
          - shape function derivatives are mapped to physical
            gradients using the inverse-transpose Jacobian.
          - a problem-dimension-dependent mass matrix M is used to
            form the tau/gamma_T estimate; the code currently
            supports 2D and a 4D hypercube variant.

        Parameters
        ----------
        fe : FiniteElement
            Finite element descriptor (must support derivative eval).
        trafo : object
            Transformation providing Jacobian and mesh information.
        intrule : IntegrationRule, optional
            Quadrature rule to use; if None a rule is chosen.

        Returns
        -------
        ndarray
            Local stabilized element vector of shape (n_local_dofs,).
        """
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
        """Compute local mass matrix for an element.

        Weak formulation (element K):
            M_{ij} = \int_K c(x) \phi_i(x) \phi_j(x) \;dx

        where c is the (possibly spatially varying) coefficient,
        phi_i are trial basis functions and phi_j are test basis
        functions.

        Parameters
        ----------
        fe_test, fe_trial : FiniteElement
            Finite element descriptors for test and trial spaces.
        trafo : object
            Transformation object providing the Jacobian mapping
            reference -> physical element and access to the mesh.
        intrule : IntegrationRule, optional
            Quadrature rule to use. If None, a rule is selected by
            polynomial orders of the elements.

        Returns
        -------
        ndarray
            Local element mass matrix of shape (n_test, n_trial).
        """
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
        """Compute local stiffness matrix for the Laplace operator.

        Weak formulation (element K):
            A_{ij} = \int_K c(x) \nabla\phi_i(x) \cdot \nabla\phi_j(x) \;dx

        This implements mapping of reference gradients to physical
        gradients using the element Jacobian and performs quadrature
        over the element.

        Parameters
        ----------
        fe_test, fe_trial : FiniteElement
            Finite element descriptors for test and trial spaces.
        trafo : object
            Transformation object providing the Jacobian mapping and
            mesh metadata.
        intrule : IntegrationRule, optional
            Quadrature rule to use. If None, a rule is selected by
            polynomial orders of the elements.

        Returns
        -------
        ndarray
            Local element stiffness matrix of shape (n_test, n_trial).
        """
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
        """Compute Laplace-like bilinear form ignoring the last dimension.

        This operator is similar to the standard Laplace integral but
        is constructed to omit the final coordinate (for example to
        exclude time when assembling spatial diffusion only). The weak
        form per element K is
            A_{ij} = \int_K c(x) \nabla_{x'}\phi_i \cdot \nabla_{x'}\phi_j \;dx

        where \nabla_{x'} denotes gradient with respect to all but
        the last coordinate.

        Parameters
        ----------
        fe_test, fe_trial : FiniteElement
            Finite element descriptors for test and trial spaces.
        trafo : object
            Transformation providing Jacobian and mesh information.
        intrule : IntegrationRule, optional
            Quadrature rule to use.

        Returns
        -------
        ndarray
            Local element matrix corresponding to the reduced Laplace
            bilinear form.
        """
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
        """Compute element matrix for time-derivative coupling.

        Implements the bilinear form corresponding to the weak
        representation of a time derivative term on the element K:
            T_{ij} = \int_K (\partial_t \phi_j) \; \phi_i \; c(x) \;dx

        In the code the trial element shape functions are evaluated
        with derivatives and the time-direction derivative is
        extracted (typically the last coordinate) then contracted with
        test shapes and quadrature weights.

        Parameters
        ----------
        fe_test, fe_trial : FiniteElement
            Finite element descriptors for test and trial spaces.
        trafo : object
            Transformation object providing Jacobian, mesh and mapping
            information.
        intrule : IntegrationRule, optional
            Quadrature rule to use.

        Returns
        -------
        ndarray
            Local element matrix representing the time-derivative
            coupling between trial and test spaces.
        """
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
        """Compute element matrix for the convection term.

        Weak formulation (element K):
            C_{ij} = \int_K (w(x) \cdot \nabla \phi_j(x)) \; \phi_i(x) \; |\det J| \; dx

        where w(x) is the convective velocity provided as `coeff`.
        The implementation maps reference gradients to physical
        gradients and forms the scalar convective derivative
        w \cdot \nabla phi_j which is then multiplied by the test
        function and integrated.

        Parameters
        ----------
        fe_test, fe_trial : FiniteElement
            Test and trial finite element descriptors.
        trafo : object
            Transformation providing Jacobians and mesh metadata.
        intrule : IntegrationRule, optional
            Quadrature rule to use.

        Returns
        -------
        ndarray
            Local convection matrix of shape (n_test, n_trial).
        """
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
        """Compute the SUPG stabilization matrix.

        The SUPG (streamline upwind Petrov-Galerkin) stabilization
        implemented here computes, per element K, a stabilizing term
            S_{ij} = \int_K \gamma_T(x) \; (w\cdot\nabla\phi_i) \; (w\cdot\nabla\phi_j) \; dx

        where gamma_T is a stabilization parameter depending on the
        local mesh size and the velocity w. The method maps reference
        gradients to physical gradients, computes the directional
        derivatives along w and assembles the weighted inner product.

        Parameters
        ----------
        fe_test, fe_trial : FiniteElement
            Test and trial finite element descriptors (both with
            derivative evaluations required).
        trafo : object
            Transformation providing Jacobians and mesh metadata.
        intrule : IntegrationRule, optional
            Quadrature rule to use.

        Returns
        -------
        ndarray
            Local SUPG stabilization matrix of shape (n_test, n_trial).
        """
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
        """Assemble the coupling matrix for div(u) against a scalar test.

        Weak formulation (element K):
            B_{ij} = \int_K q_i(x) \; (\nabla\cdot u_j(x)) \; dx

        Here q_i are scalar (pressure-like) test functions and u_j are
        vector-valued trial functions. The implementation computes
        physical gradients, forms the divergence of the vector trial
        basis and integrates against the scalar test basis.

        Parameters
        ----------
        fe_test : FiniteElement
            Scalar test element (q functions).
        fe_trial : FiniteElement
            Vector trial element (u functions) with derivatives.
        trafo : object
            Transformation providing Jacobian and mesh information.
        intrule : IntegrationRule, optional
            Quadrature rule to use.

        Returns
        -------
        ndarray
            Local coupling matrix between div(u) and q.
        """
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
        """Assemble coupling matrix for divergence of test velocity with pressure.

        Weak formulation (element K):
            B_{ij} = \int_K (\nabla\cdot v_i(x)) \; p_j(x) \; dx

        This is the transpose-like counterpart to DivUQIntegrator and is
        used when assembling mixed formulations (e.g. velocity-pressure
        coupling in Stokes/Navier–Stokes). The code computes the
        divergence of (test) vector basis functions and integrates
        against scalar pressure trial basis functions.

        Parameters
        ----------
        fe_test : FiniteElement
            Vector-valued test element (v) with derivative evaluations.
        fe_trial : FiniteElement
            Scalar pressure trial element (p).
        trafo : object
            Transformation providing Jacobian and mesh information.
        intrule : IntegrationRule, optional
            Quadrature rule to use.

        Returns
        -------
        ndarray
            Local coupling matrix between div(v) and p.
        """
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

        """Compute pressure stabilization matrix.

        Implements s(p,q) = delta * \int_K \nabla p \cdot \nabla q \; dx
        but returns the scaled and (negatively) stabilized version used
        in the code (-h^2 * s(p,q)) where h is the local mesh size.

        Parameters
        ----------
        fe_test, fe_trial : FiniteElement
            Scalar finite element descriptors for pressure test and trial.
        trafo : object
            Transformation providing Jacobian and mesh metadata.
        intrule : IntegrationRule, optional
            Quadrature rule to use.

        Returns
        -------
        ndarray
            Local stabilization matrix for the pressure space.
        """

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

