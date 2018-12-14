import numpy as np
import george as gg
import covariance_breakdown as cb
import george
from george.kernels import *
import scipy.optimize as op
import copy

class CovEmu(object):
    """
    Generalized emulator for covariance matrices.
    """
    def __init__(self, parameters, covariance_matrices, Npc_d=1, Npc_l=1):
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
        self.Npc_d = Npc_d
        self.Npc_l = Npc_l
        self.number_of_matrices  = len(Cs)
        self.matrix_size         = len(Cs[0])
        self.covariance_matrices = Cs
        self.parameters           = parameters
        if parameters.ndim == 2:
            self.Npars = len(self.parameters[0])
        else:
            self.Npars = 1 #1 number per covariance matrix

        #Call methods that start to build the emulator
        self.breakdown_matrices()
        self.create_training_data()
        self.build_emulator()
        self.train_emulator()

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
        self.ds_raw = np.log(Ds)
        self.Lprimes = Lprimes
        #Compute their first statistical moments
        self.d_mean = np.mean(self.ds_raw)
        self.d_std  = np.std(self.ds_raw)
        self.Lprime_mean = np.mean(Lprimes)
        self.Lprime_std  = np.std(Lprimes)
        #If any standard deviations are 0, set them to 1
        if self.d_std == 0:
            self.d_std = 1
        if self.Lprime_std == 0:
            self.Lprime_std = 1
        return

    def create_training_data(self):
        """
        Take the broken down matrices and create 
        training data using PCA via SVD.
        """
        Npc_d = self.Npc_d
        Npc_l = self.Npc_l
        #Regularize the broken down data        
        self.ds  = (self.ds_raw - self.d_mean)/self.d_std
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
        ws_l, phis_l = compute_ws_and_phis(self.lps, Npc_l)
        #Save the weights and PCs
        self.ws_d   = ws_d
        self.phis_d = phis_d
        self.ws_l   = ws_l
        self.phis_l = phis_l
        return

    def build_emulator(self, kernel_d=None, kernel_lp=None):
        metric_guess = np.std(self.parameters, 0)
        if kernel_d is None:
            kernel_d  = 1.*ExpSquaredKernel(metric_guess, ndim=self.Npars)
        if kernel_lp is None:
            kernel_lp = 1.*ExpSquaredKernel(metric_guess, ndim=self.Npars)

        gplist_d = []
        #Create all GPs for d; one for each principle component
        for i in range(self.Npc_d):
            ws = self.ws_d[i,:]
            kd = copy.deepcopy(kernel_d)
            gp = george.GP(kernel=kd, fit_kernel=True, mean=np.mean(ws))
            gp.compute(self.parameters)
            gplist_d.append(gp)
            continue
        gplist_l = []
        #Create all GPs for lprime; one for each principle component
        for i in range(self.Npc_l):
            ws = self.ws_l[i,:]
            kl = copy.deepcopy(kernel_lp)
            gp = george.GP(kernel=kl, fit_kernel=True, mean=np.mean(ws))
            gp.compute(self.parameters)
            gplist_l.append(gp)
            continue
        self.gplist_d  = gplist_d
        self.gplist_l  = gplist_l
        self.GPs_built = True
        return

    def train_emulator(self):
        if not self.GPs_built:
            raise Exception("Need to build before training.")
        
        #Train the GPs for d
        for i, gp in enumerate(self.gplist_d):
            ws = self.ws_d[i,:]
            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(ws, quiet=True)
                return -ll if np.isfinite(ll) else 1e25
            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(ws, quiet=True)
            p0 = gp.get_parameter_vector()
            result = op.minimize(nll, p0, jac=grad_nll)
            gp.set_parameter_vector(result.x)
            continue
        #Train the GPs for lprime
        for i, gp in enumerate(self.gplist_l):
            ws = self.ws_l[i,:]
            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(ws, quiet=True)
                return -ll if np.isfinite(ll) else 1e25
            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(ws, quiet=True)
            p0 = gp.get_parameter_vector()
            result = op.minimize(nll, p0, jac=grad_nll)
            gp.set_parameter_vector(result.x)
            continue
        self.trained = True
        return

    def predict(self, params):
        if not self.trained:
            raise Exception("Need to train the emulator first.")

        params = np.atleast_1d(params)
        if params.ndim > 1:
            raise Exception("'params' must be a single point in parameter "+
                            "space; a 1D array at most.")
        if len(params) != self.Npars:
            raise Exception("length of 'params' does not match training "+\
                            "parameters.")
        #Loop over d GPs and predict weights
        wp_d = np.array([gp.predict(ws, params)[0] for ws,gp in zip(self.ws_d, self.gplist_d)])
        wp_l = np.array([gp.predict(ws, params)[0] for ws,gp in zip(self.ws_l, self.gplist_l)])
        #Multiply by the PCs to get predicted ds and lprimes
        d_pred  = wp_d[0]*self.phis_d[0]
        lp_pred = wp_l[0]*self.phis_l[0]
        
        for i in range(1,self.Npc_d):
            d_pred  += wp_d[i]*self.phis_d[i]

        for i in range(1,self.Npc_l):
            lp_pred += wp_l[i]*self.phis_l[i]

        #Multiply on the stddev and add on the mean
        d_pred_raw  = d_pred *self.d_std + self.d_mean
        Lprime_pred = lp_pred*self.Lprime_std + self.Lprime_mean
        D_pred = np.exp(d_pred_raw)
        #Reconstruct the covariance through the breakdown tool
        breakdown_predicted = cb.breakdown.from_D_Lprime(D_pred, Lprime_pred)
        return breakdown_predicted.C
