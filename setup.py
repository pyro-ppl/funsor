from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

setup(
    name='funsor',
    version='0.0.0',
    description='Functional analysis + tensors + symbolic algebra',
    packages=find_packages(include=['funsor', 'funsor.*']),
    url='https://github.com/pyro-ppl/funsor',
    author='Uber AI Labs',
    author_email='fritzo@uber.com',
    install_requires=[
        'contextlib2',
        'multipledispatch',
        'numpy>=1.7',
        'opt_einsum>=2.3.2',
        'pyro-ppl>=0.3',
        'six>=1.10.0',
        'torch>=1.0.0',
        'unification',
    ],
    extras_require={
        'test': ['flake8', 'pytest>=4.1', 'torchvision==0.2.1'],
        'dev': [
            'flake8',
            'isort',
            'pytest>=4.1',
            'sphinx>=2.0',
            'sphinx_rtd_theme',
            'torchvision==0.2.1',
        ],
    },
    tests_require=['flake8', 'pytest>=4.1'],
    keywords='probabilistic machine learning bayesian statistics pytorch',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
    ],
)
