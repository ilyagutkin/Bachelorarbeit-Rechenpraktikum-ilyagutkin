import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, bicgstab ,gmres
from scipy.sparse.linalg import svds

def solve_on_freedofs(A, b, free_dofs, total_dofs=None):

    if total_dofs is None:
        total_dofs = A.shape[0]

    # Reduzierte Matrix und rechte Seite extrahieren
    A_reduced = A[free_dofs, :][:, free_dofs]
    b_reduced = b[free_dofs]
    # Lösen des reduzierten Systems
    x_reduced = spsolve(A_reduced, b_reduced)

    # Ergebnis auffüllen
    x_full = np.zeros(total_dofs)
    x_full[free_dofs] = x_reduced

    return x_full