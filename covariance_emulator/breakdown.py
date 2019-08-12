import numpy as np

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

        #Error checking
        if C.ndim < 2:
            raise Exception("Covariance matrix has too few dimensions.")
        if C.ndim > 2:
            raise Exception("Covariance matrix has too many dimensions.")
        if not np.allclose(C, C.T, atol=1e-8):
            raise Exception("Covariance matrix is not symmetric.")

        #Save the covariance
        self.C = C

        #Perform GCD
        #will be slow if C is large
        #in which case swap this for a better library
        Lch = np.linalg.cholesky(C)
        S = np.diag(np.diag(Lch))
        Sinv = np.linalg.inv(S)
        D = np.diag(np.dot(S,S))
        L = np.dot(Lch,Sinv)
        self.D = D
        self.L = L

        #Loop over the independent elements of L and save them
        ND = len(D)
        Lprime = np.zeros(int(ND*(ND-1)/2))
        k = 0
        for i in range(1, ND):
            if not unravel_diagonally:
                for j in range(0,i):
                    Lprime[k] = L[i, j]
                    k+=1
            else:
                for j in range(0, ND-i):
                    Lprime[k] = L[i+j, j]
                    k+=1
        self.Lprime = Lprime

        #Also save the eiegenvalues and rotation matrix
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
        if not int((ND*(ND-1)/2) == len(Lprime)):
            raise Exception("Mismatched length:\n\tlen(Lprime) must be len(D)*(len(D)-1)/2")
        
        L = np.zeros((ND, ND))
        k = 0
        for i in range(1, ND):
            if not unravel_diagonally:
                for j in range(0, i):
                    L[i, j] = Lprime[k]
                    k+=1
            else:
                for j in range(0, ND-i):
                    L[i+j, j] = Lprime[k]
                    k+=1
        
        for i in range(0, ND):
            L[i,i] = 1.
        C = np.dot(L, np.dot(np.diag(D), L.T))
        return cls(C, unravel_diagonally)
