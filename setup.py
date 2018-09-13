#!/usr/bin/env python
"""diagnostic setup."""
import setuptools


setuptools.setup(name='diagnostic',
                 version='0.0.0',
                 author='Edward Higson',
                 install_requires=['nestcheck>=0.1.6',
                                   'dyPolyChord>=0.0.2',
                                   'numpy',
                                   'scipy',
                                   'pandas',
                                   'getdist',
                                   'matplotlib',
                                   'more_itertools'],
                 packages=['diagnostic'])
