import numpy as np
from methodsnm.fe import FE     

class BlockFE(FE):
    """
    Finite Element for product spaces (Vector FE).
    Holds several scalar FEs and concatenates their shape functions.
    """

    def __init__(self, fe_list):
        self.fes = fe_list
        self.nblocks = len(fe_list)
        self.order = max(fe.order for fe in fe_list)
        self.block_ndofs = [fe.ndof for fe in fe_list]
        self.block_offsets = [0]

        for nd in self.block_ndofs:
            self.block_offsets.append(self.block_offsets[-1] + nd)

        self.ndof = self.block_offsets[-1]

    def _evaluate_id(self, ip):
        phis = [fe.evaluate(ip) for fe in self.fes]
        return np.concatenate(phis, axis=-1)

    def _evaluate_deriv(self, ip, direction=None):
        """Evaluate derivatives at ip -> shape (dim, ndof_total)."""
        if direction is None:
            grads = [fe._evaluate_deriv(ip) for fe in self.fes]
        else:
            grads = [fe._evaluate_deriv(ip, direction) for fe in self.fes]
        return np.concatenate(grads, axis=-1)
