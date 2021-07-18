"""Emulator for covariance matrices.

This module provides an emulator for covariance matrices, or a tool
to predict covariance matrices in a domain space given a set of
training points in that domain space.

"""
from .covariance_emulator import CovEmu

assert CovEmu  # A hack to get pyflakes to not complain

__version__ = "0.1.0"
__author__ = "Thomas McClintock"
__email__ = "thmsmcclintock@gmail.com"
__docs__ = "Emulate parameter dependent covariance matrices."