# nestcheck: diagnostic tests for nested sampling calculations

[![arXiv](http://img.shields.io/badge/arXiv-1804.06406-B31B1B.svg)](https://arxiv.org/abs/1804.06406)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ejhigson/dns/blob/master/LICENSE)

This repository contains the code used for making the results and plots in "nestcheck: diagnostic tests for nested sampling calculations" ([Higson et. al, 2018](https://arxiv.org/abs/1804.06406)). This provides examples of the use of the ``nestcheck`` package.

If you have any questions then feel free to email <e.higson@mrao.cam.ac.uk>. However, note that this is research code and is not actively maintained.

### Requirements

Generating the results in the paper requires ``PolyChord`` v1.14, plus the requirements listed in ``setup.py``. Results in the paper were run using Python 3.6, ``nestcheck`` v0.1.6 and ``PolyChord`` v1.14.

### Install

The ``diagnostic`` Python module contains high level functions for generating and plotting the results in the paper. Most of this is just convenient wrappers and stored settings for using the ``nestcheck`` module, which contains implementations of the tests introduced in the paper. ``diagnostic`` can be installed, along with ``nestcheck`` and its other dependencies, by running the following command from within this repo:

```
pip install . --user
```

### Generating nested sampling runs

``generate_data.py`` can be used to generate the nested sampling runs used in the paper except those using data from the *Planck* survey - see its documentation for more details. For details of the likelihood used for the *Planck* survey and how it can be downloaded see the paper and references therein.

### Making the paper plots

All the plots in the paper can be generated using the ``diagnostic_paper_code.ipynb`` and ``planck_results_plotting.ipynb`` notebooks; see the notes within each for more details.

### Attribution

If you use this code in your academic research, please cite the ``nestcheck`` references. These are listed in the attribution section of the documentation at <https://nestcheck.readthedocs.io/en/latest/>.
