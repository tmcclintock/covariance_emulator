import os

PATH_ROOT = os.path.dirname(__file__)
from setuptools import setup

import covariance_emulator  # noqa: E402

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()


dist = setup(
    name="covariance_emulator",
    author="Thomas McClintock",
    author_email="thmsmcclintock@gmail.com",
    license="MIT",
    url="https://github.com/tmcclintock/covariance_emulator",
    packages=["covariance_emulator"],
    long_description=open("README.md").read(),
    version=covariance_emulator.__version__,
    description=covariance_emulator.__docs__,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
