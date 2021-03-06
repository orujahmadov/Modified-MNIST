from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['keras>=2.0.8','sklearn>=0.0', 'h5py>=2.7.0', 'google-cloud-storage>=1.5.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Modified MNIST'
)
