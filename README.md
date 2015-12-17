[![Build Status](https://travis-ci.org/tylerjereddy/diffusion_analysis_MD_simulations.svg?branch=master)](https://travis-ci.org/tylerjereddy/diffusion_analysis_MD_simulations)
[![Coverage Status](https://coveralls.io/repos/tylerjereddy/diffusion_analysis_MD_simulations/badge.svg?branch=master&service=github)](https://coveralls.io/github/tylerjereddy/diffusion_analysis_MD_simulations?branch=master)
[![Documentation Status](https://readthedocs.org/projects/diffusion-analysis-md-simulations/badge/?version=latest)](http://diffusion-analysis-md-simulations.readthedocs.org/en/latest/?badge=latest)
                

Analysis of particle diffusion in molecular dynamics simulations
================================================================

Contains Python utility functions for the analysis of diffusion in molecular dynamics simulation trajectories.

The documentation for the project is available here: http://diffusion-analysis-md-simulations.readthedocs.org/en/latest/index.html

An IPython notebook containing an example diffusion calculation is available: https://github.com/tylerjereddy/diffusion_analysis_MD_simulations/blob/master/diffusion_analysis_sim126_extended.ipynb 

Please cite: [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.11827.png)](http://dx.doi.org/10.5281/zenodo.11827)

For contributions:
  * ensure all unit tests pass (run nosetests)
  * ensure all doctests pass (run doctesting.py)
  * if you import new modules, you may need to mock them in the Sphinx conf.py documentation file so that the docs are properly compiled by readthedocs
  * attempt to match the [numpy documentation standard](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) as closely as possible
