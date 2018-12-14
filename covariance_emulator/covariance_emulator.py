import numpy as np
import george as gg
import covariance_breakdown as cb
from george.kernels import *
import scipy.optimize as op

class CovEmu(object):
    """
    Generalized emulator for covariance matrices.
    """
    def __init__(self, parameters, covariance_matrices):
        parameters = np.array(parameters)
        Cs = np.array(covariance_matrices)
        #Check dimensionality
        if len(parameters) != len(Cs):
            raise Exception("Must supply a parameter (list) for each "+\
                            "covariance matrix.")
        if parameters.ndim > 2:
            raise Exception("'parameters' must be either a 1D or 2D array.")
        if Cs.ndim != 3:
            raise Exception("Must supply a list of 2D covariance matrices.")
        for i in range(0,len(Cs)-1):
            if Cs[i].shape != Cs[i+1].shape:
                raise Exception("All covariances must have the "+\
                                "same dimensions.")
            continue
        #Save all attributes
        self.number_of_matrices  = len(Cs)
        self.matrix_size         = len(Cs[0])
        self.covariance_matrices = Cs
        self.parametrs           = parameters
        if parameters.ndim == 2:
            self.Npars = len(self.parameters[0])
        else:
            self.Npars = 1 #1 number per covariance matrix

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
        #Save the broken down data
        self.Ds = Ds
        self.Lprimes = Lprimes
        #Compute their first statistical moments
        self.D_mean = np.mean(Ds)
        self.D_std  = np.std(Ds)
        self.Lprime_mean = np.mean(Lprimes)
        self.Lprime_std  = np.std(Lprimes)
        return

    def create_training_data(self, Npc_d=6, Npc_l=6):
        """
        Take the broken down matrices and create 
        training data using PCA via SVD.
        """
        #Regularize the broken down data
        self.ds  = (self.Ds - self.D_mean)/self.D_std
        self.lps = (self.Lprimes - self.Lprime_mean)/self.Lprime_std
        #Perform PCA to create weights and principle components
        def compute_ws_and_phis(A, Npc):
            u,s,v = np.linalg.svd(A, 0) #Do the PCA
            s = np.diag(s)
            N = len(s)
            P = np.dot(v.T, s)/np.sqrt(N)
            phis = P.T[:Npc]
            ws = np.sqrt(N) * u.T[:Npc]
            return ws, phis
        ws_d, phis_d = compute_ws_and_phis(self.ds, Npc_d)
        ws_l, phis_l = compute_ws_and_phis(self.ds, Npc_l)
        #Save the weights and PCs
        self.ws_d   = ws_d
        self.phis_d = phis_d
        self.ws_l   = ws_l
        self.phis_l = phis_l
        return

    def train_GPs(self, kernel_d=None, kernel_lp=None):
        gplist_d = []
        gplist_l = []
        Npars = self.Npars
        metric_guess = np.std(self.parameters, 0)
        if kernel_d is None:
            kernel_d  = 1.*ExpSquaredKernel(metric_guess, ndim=Npars)
        if kernel_lp is None:
            kernel_lp = 1.*ExpSquaredKernel(metric_guess, ndim=Npars)
        #Took a break here
