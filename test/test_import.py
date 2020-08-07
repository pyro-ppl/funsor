# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
from importlib import import_module

from funsor import get_backend


def test_all_modules_are_imported():
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'funsor')
    for path in glob.glob(os.path.join(root, '*.py')):
        name = os.path.basename(path)[:-3]
        if name in ({'torch', 'jax'} - {get_backend()}):
            assert not hasattr(import_module('funsor'), name)
            continue
        if name.startswith('__'):
            continue
        if name == "minipyro":
            continue  # TODO: enable when minipyro is backend-agnostic
        assert hasattr(import_module('funsor'), name), 'funsor/__init__.py does not import {}'.format(name)
        actual = getattr(import_module('funsor'), name)
        expected = import_module('funsor.{}'.format(name))
        assert actual == expected
