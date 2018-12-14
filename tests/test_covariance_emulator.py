import covariance_emulator as ce
import numpy as np
import numpy.testing as npt

def test_emulator_builds():
    params = np.arange(10)
    Cs = np.array([np.identity(2) for _ in params])
    Emu = ce.CovEmu(params, Cs)
    return

def test_attributes():
    attrs = ["number_of_matrices", "matrix_size", "covariance_matrices",
             "parameters", "Npars"]
    params = np.arange(10)
    Cs = np.array([np.identity(2) for _ in params])
    Emu = ce.CovEmu(params, Cs)
    for attr in attrs:
        assert hasattr(Emu, attr), True
    return
