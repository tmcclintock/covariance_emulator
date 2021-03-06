import covariance_emulator as ce
import numpy as np
import numpy.testing as npt

def test_emulator_builds():
    params = np.arange(10)
    Cs = np.array([np.identity(2) for _ in params])
    Emu = ce.CovEmu(params, Cs)
    assert Emu #builds
    return

def test_emulator_attributes():
    attrs = ["covariance_matrices", "parameters", "Npars", "Ds", "Lprimes", "d_mean", "d_std", "Lprime_mean", "Lprime_std", "ds", "lps", "ws_d", "ws_l", "phis_d", "phis_l", "NPC_D", "NPC_L", "gplist_d", "gplist_l", "kernel_D", "kernel_lp"]
    params = np.arange(1,10)
    Cs = np.array([p*np.identity(2) for p in params])
    Emu = ce.CovEmu(params, Cs)
    for attr in attrs:
        npt.assert_equal(hasattr(Emu, attr), True, err_msg="attribute %s not present"%attr)
    return

def test_emulator_predicts():
    params = np.arange(1,10)
    Cs = np.array([p*np.identity(2) for p in params])
    Emu = ce.CovEmu(params, Cs)
    Cp = Emu.predict(4.5)
    assert Cp is not None

    pars2 = np.array([params, params]).T
    Cs = np.array([p*np.identity(2) for p in params])
    Emu = ce.CovEmu(pars2, Cs)
    Cp = Emu.predict([4.5, 4.5])
    assert Cp is not None

    return


def test_emulator_exceptions():
    
    params = np.arange(1,10)
    Cs = np.array([p*np.identity(2) for p in params])
    #Wrong number of parameters vs number of covariance matrices
    with npt.assert_raises(Exception):
        Emu = ce.CovEmu(params[:-1], Cs)
    #Params is too high dimsensional
    with npt.assert_raises(Exception):
        Emu = ce.CovEmu(np.array([[params]]), Cs)
    #Covariance matrix list has wrong dimensionality (!=3)
    with npt.assert_raises(Exception):
        Emu = ce.CovEmu(params, [Cs])
    with npt.assert_raises(Exception):
        Emu = ce.CovEmu(params, Cs[0])
    #One covariance matrix has the wrong shape
    with npt.assert_raises(Exception):
        Ctemp = [p*np.identity(2) for p in params]
        Ctemp[0] = Ctemp[0][:-1]
        Emu = ce.CovEmu(params, Ctemp)
    #Prediction params are wrong size
    Emu = ce.CovEmu(params, Cs)
    with npt.assert_raises(Exception):
        Emu.predict([0,1])
    #Prediction params are wrong dimension
    with npt.assert_raises(Exception):
        Emu.predict([[0,1]])
    return
