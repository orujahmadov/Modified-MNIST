from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['keras=2.0.6','theano=0.9.0','h5py=2.7.0']

setup(
    name='trainer',
    version='0.1',
    install_required='REQUIRED_PACKAGES',
    packages=find_packages(),
    include_package_data=True,
    description='Modified MNIST',
)
