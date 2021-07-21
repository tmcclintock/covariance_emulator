"""
Methods for decomposing a covariance matrix into more fundamental components.
"""

from typing import Dict

import numpy as np


def breakdown_covariance(C: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Docompose a covariance matrix into all possible components.

    Args:
      C (np.ndarray): a square, real symmetric matrix. In the context
        of this project, this is a covariance matrix for a MVN distribution.

    Returns:
      dictionary of different decompositions, including the LDLT cholesky
      decomposition and an eigenvalue decomposition
    """
    assert C.ndim == 2
    assert len(C) == len(C[0]), "matrix not square"
    np.testing.assert_allclose(C, C.T, err_msg="covariance must match transpose")

    # LDLT Cholesky decompose
    Lch = np.linalg.cholesky(C)

    # Separate the diagonal (S) from the off-diagonal (Lprime)
    S = np.diag(np.diag(Lch))
    Sinv = np.linalg.inv(S)
    L = Lch @ Sinv
    D = np.diag(np.dot(S, S))

    # Lprime is L without the diagonal, flattened
    Lprime = L[np.tril_indices_from(L, k=-1)]

    # Also save the eiegenvalues and rotation matrix
    eigenvalues, rotation_matrix = np.linalg.eig(C)

    return {
        "C": C,
        "S": S,
        "Sinv": Sinv,
        "Lch": Lch,
        "L": L,
        "D": D,
        "Lprime": Lprime,
        "eigenvalues": eigenvalues,
        "rotation_matrix": rotation_matrix,
    }


def breakdown_covariance_from_components(
    D: np.ndarray,
    Lprime: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Docompose a covariance matrix into all possible components,
    starting from the ``D`` and ``Lprime`` components that we
    emulate. This is a helper function for our emulation strategy.

    Args:
      D (np.ndarray): diagonal components
      Lprime (np.ndarray): elements of the lower-triangular matrix
        from the LDLT Cholesky decomposition

    Returns:
      dictionary of different decompositions, including the LDLT cholesky
      decomposition and an eigenvalue decomposition
    """
    assert Lprime.ndim == 1

    # Promote D to 2D if it isn't
    D = np.diag(D) if D.ndim == 1 else D

    # Reconstruct L then compute C = LDLT
    L: np.ndarray = np.eye(len(D))
    L[np.tril_indices_from(L, k=-1)] = Lprime
    C: np.ndarray = L @ (D @ L.T)
    return breakdown_covariance(C)
