# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import sys

from setuptools import find_packages, setup

# READ README.md for long description on PyPi.
# This requires uploading via twine, e.g.:
# $ python setup.py sdist bdist_wheel
# $ twine upload --repository-url https://test.pypi.org/legacy/ dist/*  # test version
# $ twine upload dist/*
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to convert README.md to rst:\n  {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

# Remove badges since they will always be obsolete.
# This assumes the first 4 lines contain badge info.
long_description = "\n".join(line for line in long_description.split("\n")[4:])

setup(
    name="funsor",
    version="0.4.5",  # mirrored in funsor/__init__.py
    description="A tensor-like library for functions and distributions",
    packages=find_packages(include=["funsor", "funsor.*"]),
    url="https://github.com/pyro-ppl/funsor",
    project_urls={"Documentation": "https://funsor.pyro.ai"},
    author="Uber AI Labs",
    python_requires=">=3.7",
    install_requires=[
        "makefun",
        "multipledispatch",
        "numpy>=1.7",
        "opt_einsum>=2.3.2",
        "typing_extensions",
    ],
    extras_require={
        "torch": ["pyro-ppl>=1.8.0", "torch>=1.11.0"],
        "jax": ["numpyro>=0.7.0", "jax>=0.2.21", "jaxlib>=0.1.71"],
        "test": [
            "black",
            "flake8",
            "isort>=5.0",
            "pandas",
            "pillow==8.2.0",  # https://github.com/pytorch/pytorch/issues/61125
            "pyro-api>=0.1.2",
            "pytest==4.3.1",
            "pytest-xdist==1.27.0",
            "requests",
            "scipy",
            "torchvision>=0.12.0",
        ],
        "dev": [
            "black",
            "flake8",
            "isort>=5.0",
            "nbsphinx",
            "pandas",
            "pillow==8.2.0",  # https://github.com/pytorch/pytorch/issues/61125
            "pytest==4.3.1",
            "pytest-xdist==1.27.0",
            "scipy",
            "sphinx>=2.0",
            "sphinx-gallery",
            "sphinx_rtd_theme",
            "torchvision>=0.12.0",
        ],
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="probabilistic machine learning bayesian statistics pytorch jax",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",  # for jax but not torch
    ],
)
