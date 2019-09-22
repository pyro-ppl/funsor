import glob
import os
from importlib import import_module


def test_all_modules_are_imported():
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'funsor')
    for path in glob.glob(os.path.join(root, '*.py')):
        name = os.path.basename(path)[:-3]
        if name.startswith('__'):
            continue
        assert hasattr(import_module('funsor'), name), f'funsor/__init__.py does not import {name}'
        actual = getattr(import_module('funsor'), name)
        expected = import_module(f'funsor.{name}')
        assert actual == expected
