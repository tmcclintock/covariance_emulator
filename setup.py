from setuptools import setup

dist = setup(name="covariance_emulator",
             author="Thomas McClintock",
             author_email="mcclintock@bnl.gov",
             description="Framework for emulating covariance matrices.",
             license="MIT",
             url="https://github.com/tmcclintock/covariance_emulator",
             packages=['covariance_emulator'],
             long_description=open("README.md").read()
)
