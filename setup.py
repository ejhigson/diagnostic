#!/usr/bin/env python
"""diagnostic setup."""
import setuptools


setuptools.setup(name='diagnostic',
                 version='0.0.0',
                 author='Edward Higson',
                 install_requires=['nestcheck',
                                   'dyPolyChord',
                                   'more_itertools'],
                 packages=['diagnostic'])
