from setuptools import setup, find_packages

setup(
    name="ftle_package",
    version="0.1.0",
    description="A Python package for computing FTLE fields in flat and curved sub domains of Euclidean 3 space",
    author="Blase Fencil",
    author_email="bafencil@gmail.com",
    packages=find_packages(),  
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyvista",
        "numba"
    ],
    python_requires=">=3.7",
)
