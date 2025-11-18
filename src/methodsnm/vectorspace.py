import numpy as np
from methodsnm.fes import FESpace
from methodsnm.fe import FE
from methodsnm.fe_vector import BlockFE

class Productspace(FESpace):
    def __init__(self, spaces):
        self.spaces = spaces
        self.mesh = spaces[0].mesh
        self.ndof = sum(space.ndof for space in spaces)
        self.offsets = [0]
        for V in spaces:
            self.offsets.append(self.offsets[-1] + V.ndof)     

    def component_space(self, i):
        return self.spaces[i]

    def block_range(self, b):
        return slice(self.offsets[b], self.offsets[b+1])
    
    def blocks(self):
        return [self.block(i) for i in range(self.space.nblocks)]

    def _finite_element(self, elnr):
        fe_list = [V.finite_element(elnr) for V in self.spaces]
        return BlockFE(fe_list)
    
    def _element_dofs(self, elnr):
        dofs_list = []
        for b, V in enumerate(self.spaces):
            local = V.element_dofs(elnr)
            offset = self.offsets[b]
            dofs_list.append(local + offset)
        return np.concatenate(dofs_list)
    
    def get_freedofs(self, blocked = None):
        """
        Return the global free DOF indices of the product space.

        Parameters
        ----------
        blocked : dict
            Dictionary mapping block indices to lists of locally blocked (Dirichlet)
            DOF indices. These local indices are shifted by the block offset to
            obtain global blocked DOFs.

        Returns
        -------
        ndarray
            Array of global free DOF indices. A boolean mask is used internally
            because it allows efficient marking and slicing of DOFs in FEM systems.
        """
        if blocked is None:
            blocked = {}
        gmask = np.ones(self.ndof, dtype=bool)
        for b, idxs in blocked.items():
            off = self.offsets[b]
            gmask[off + np.array(idxs, dtype=int)] = False
        return np.where(gmask)[0]
