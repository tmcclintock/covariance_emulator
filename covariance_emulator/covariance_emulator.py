import numpy as np
import george as gg
import covariance_breakdown as cb

class CovEmu(object):
    """
    Generalized emulator for covariance matrices.
    """
    def __init__(self, covariance_matrices):
        Cs = np.array(covariance_matrices)
        #Check dimensionality
        if Cs.ndim != 3:
            raise Exception("Must supply a list of 2D covariance matrices.")
        for i in range(0,len(Cs)-1):
            if Cs[i].shape != Cs[i+1].shape:
                raise Exception("All covariances must have the "+\
                                "same dimensions.")
            continue
        #Save
        self.number_of_matrices  = len(Cs)
        self.matrix_size         = len(Cs[0])
        self.covariance_matrices = Cs

    @classmethod
    def from_Ds_Lprimes(cls, Ds, Lprimes):
        """
        Reconstruct all covariance matrices from their individual parts
        and assemble the emulator from those.
        """
        pass

    def breakdown_matrices(self):
        """
        Break down matrices into their constituent parts.
        :returns:
            None
        """
        Cs  = self.covariance_matrices
        ND  = self.matrix_size
        Nc  = self.number_of_matrices
        NLp = int(ND*(ND-1)/2)
        Ds  = np.zeros((Nc, ND))
        Lprimes = np.zeros((Nc,NLp))
        #Loop over matrices and break them down
        for i in range(Nc):
            b          = cb.breakdown(Cs[i])
            Ds[i]      = b.D
            Lprimes[i] = b.Lprime
            continue
        self.Ds = Ds
        self.Lprimes = Lprimes
        return
