"""
Tests of the functions in breakdown.py.
"""
from unittest import TestCase

import numpy as np
import numpy.testing as npt
from scipy.stats import invwishart

from covariance_emulator.breakdown import (
    breakdown_covariance,
    breakdown_covariance_from_components,
)


class BreakdownTest(TestCase):
    @staticmethod
    def assert_reconstruction(c, outdict=None):
        if outdict is None:
            outdict = breakdown_covariance(c)
            assert outdict["C"] is c
        Lch = outdict["Lch"]
        npt.assert_allclose(Lch @ Lch.T, c)
        D = outdict["D"]
        L = outdict["L"]
        D = np.diag(D)
        npt.assert_allclose(L @ D @ L.T, c)
        w = outdict["eigenvalues"]
        v = outdict["rotation_matrix"]
        npt.assert_allclose(v @ np.diag(w) @ v.T, c, rtol=1e-5)

    @classmethod
    def assert_reconstruction_from_breakdown(cls, c):
        outdict = breakdown_covariance(c)
        D = outdict["D"]
        Lprime = outdict["Lprime"]
        outdict = breakdown_covariance_from_components(D, Lprime)
        cls.assert_reconstruction(c, outdict)

    def test_small_matrix(self):
        c = invwishart.rvs(df=3, scale=[1, 10, 100])
        self.assert_reconstruction(c)

    def test_many_medium_matrices(self):
        df = 200
        for _ in range(100):
            scale = np.random.rand(df)
            c = invwishart.rvs(df=df, scale=scale)
            self.assert_reconstruction(c)

    def test_large_matrix(self):
        df = 900
        scale = np.random.rand(df)
        c = invwishart.rvs(df=df, scale=scale)
        self.assert_reconstruction(c)

    def test_small_matrix_from_components(self):
        c = invwishart.rvs(df=3, scale=[1, 10, 100])
        self.assert_reconstruction_from_breakdown(c)

    def test_large_matrix_from_components(self):
        df = 900
        scale = np.random.rand(df)
        c = invwishart.rvs(df=df, scale=scale)
        self.assert_reconstruction_from_breakdown(c)

    def test_many_medium_matrices_from_components(self):
        df = 200
        for _ in range(100):
            scale = np.random.rand(df)
            c = invwishart.rvs(df=df, scale=scale)
            self.assert_reconstruction_from_breakdown(c)
