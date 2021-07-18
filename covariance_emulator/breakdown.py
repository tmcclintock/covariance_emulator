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


# TODO - deprecate this class
class breakdown(object):
    """
    Take in a covariance matrix and break it down into constituent parts.
    :param C:
        2D array of a covariance matrix
    :param unravel_diagonally:
        boolean flag indicating how the L submatrix is unraveled (i.e. along diagonals or not)
    """

    def __init__(self, C, unravel_diagonally=False):
        C = np.array(C)

        # Error checking
        if C.ndim < 2:
            raise Exception("Covariance matrix has too few dimensions.")
        if C.ndim > 2:
            raise Exception("Covariance matrix has too many dimensions.")
        if not np.allclose(C, C.T, atol=1e-8):
            raise Exception("Covariance matrix is not symmetric.")

        # Save the covariance
        self.C = C

        # Perform GCD
        # will be slow if C is large
        # in which case swap this for a better library
        Lch = np.linalg.cholesky(C)
        S = np.diag(np.diag(Lch))
        Sinv = np.linalg.inv(S)
        D = np.diag(np.dot(S, S))
        L = np.dot(Lch, Sinv)
        self.D = D
        self.L = L

        # Loop over the independent elements of L and save them
        ND = len(D)
        Lprime = np.zeros(int(ND * (ND - 1) / 2))
        k = 0
        for i in range(1, ND):
            if not unravel_diagonally:
                for j in range(0, i):
                    Lprime[k] = L[i, j]
                    k += 1
            else:
                for j in range(0, ND - i):
                    Lprime[k] = L[i + j, j]
                    k += 1
        self.Lprime = Lprime

        # Also save the eiegenvalues and rotation matrix
        eigenvalues, rotation_matrix = np.linalg.eig(self.C)
        self.eigenvalues = eigenvalues
        self.rotation_matrix = rotation_matrix

    @classmethod
    def from_D_Lprime(cls, D, Lprime, unravel_diagonally=False):
        """
        Reconstruct a covariance matrix from a diagonal and flattened L matrix.
        The covariance C and L matrices will be self-assigned
        and aren't returned.

        :param D:
            diagonal of decomposed covariance matrix
        :param Lprime:
            flattened lower triangular matrix from decomposition
        :param unravel_diagonally:
            boolean flag indicating how the L submatrix is unraveled (i.e. along diagonals or not)
        :return:
            None
        """
        D = np.array(D)
        Lprime = np.array(Lprime)
        if D.ndim > 1 or D.ndim == 0:
            raise Exception("D must be a 1D array")
        ND = len(D)
        if not int((ND * (ND - 1) / 2) == len(Lprime)):
            raise Exception(
                "Mismatched length:\n\tlen(Lprime) must be len(D)*(len(D)-1)/2"
            )

        L = np.zeros((ND, ND))
        k = 0
        for i in range(1, ND):
            if not unravel_diagonally:
                for j in range(0, i):
                    L[i, j] = Lprime[k]
                    k += 1
            else:
                for j in range(0, ND - i):
                    L[i + j, j] = Lprime[k]
                    k += 1

        for i in range(0, ND):
            L[i, i] = 1.0
        C = np.dot(L, np.dot(np.diag(D), L.T))
        return cls(C, unravel_diagonally)
