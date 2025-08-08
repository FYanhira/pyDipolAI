from setuptools import setup, find_packages

setup(
    name='dielectric_model_gui',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'lmfit',
        'tk',
    ],
    author='Yanhira Rentería',
    description='A GUI-based tool for fitting dielectric models to experimental data',
)
