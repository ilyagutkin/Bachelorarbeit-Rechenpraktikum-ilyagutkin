import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, bicgstab ,gmres
from scipy.sparse.linalg import svds

def solve_on_freedofs(A, b, free_dofs, total_dofs=None):
    """Solve a linear system for a subset of free DOFs and return full vector.

    This convenience routine extracts the submatrix and sub-vector
    corresponding to ``free_dofs``, solves the reduced linear system
    using SciPy's direct solver, and returns a full-length solution
    vector where constrained DOFs are zero (or left at their initial
    value).

    Parameters
    ----------
    A : (N, N) sparse matrix
        Global system matrix (can be sparse CSR/CSC).
    b : (N,) array_like
        Global right-hand side vector.
    free_dofs : array_like of int
        Indices of DOFs to solve for (remaining DOFs are treated as fixed).
    total_dofs : int, optional
        Size of the full system; if None it is inferred from A.shape[0].

    Returns
    -------
    x_full : ndarray
        Full-length solution vector with entries filled at `free_dofs`.
    """

    if total_dofs is None:
        total_dofs = A.shape[0]

    # extract reduced matrix and RHS
    A_reduced = A[free_dofs, :][:, free_dofs]
    b_reduced = b[free_dofs]

    # solve reduced system
    x_reduced = spsolve(A_reduced, b_reduced)

    # populate full solution vector
    x_full = np.zeros(total_dofs)
    x_full[free_dofs] = x_reduced

    return x_full