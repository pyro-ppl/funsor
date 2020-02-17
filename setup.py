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
    long_description = open('README.md', encoding='utf-8').read()
except Exception as e:
    sys.stderr.write(f'Failed to convert README.md to rst:\n  {e}\n')
    sys.stderr.flush()
    long_description = ''

# Remove badges since they will always be obsolete.
# This assumes the first 4 lines contain badge info.
long_description = '\n'.join(line for line in long_description.split('\n')[4:])

setup(
    name='funsor',
    version='0.1.2',
    description='A tensor-like library for functions and distributions',
    packages=find_packages(include=['funsor', 'funsor.*']),
    url='https://github.com/pyro-ppl/funsor',
    project_urls={
        "Documentation": "https://funsor.pyro.ai",
    },
    author='Uber AI Labs',
    author_email='fritzo@uber.com',
    python_requires=">=3.6",
    install_requires=[
        'multipledispatch',
        'numpy>=1.7',
        'opt_einsum>=2.3.2',
        # TODO: remove pyro-ppl and torch when funsor.minipyro and funsor.pyro is refactored
        'pyro-ppl>=0.5',
        'pytest>=4.1',
        'torch>=1.3.0',
    ],
    extras_require={
        'torch': [
            'pyro-ppl>=0.5',
            'torch>=1.3.0',
        ],
        'jax': [
            'jax==0.1.59',
            'jaxlib==0.1.38',
            'numpyro @ git+https://github.com/google/jax.git@f2aefcf#egg=numpyro'
        ],
        'test': [
            'flake8',
            'pandas',
            'pyro-api>=0.1',
            'pytest-xdist==1.27.0',
            'pillow-simd',
            'scipy',
            'torchvision',
        ],
        'dev': [
            'flake8',
            'isort',
            'pandas',
            'pytest-xdist==1.27.0',
            'scipy',
            'sphinx>=2.0',
            'sphinx_rtd_theme',
            'pillow-simd',
            'torchvision',
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='probabilistic machine learning bayesian statistics pytorch',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache License 2.0',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
    ],
)
