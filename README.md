# KernelDensityBoundaries1D class for sklearn

A custom Kernel Density Estimator class with boundary condition algorithms
compatible with scikit-learn.

### Motivation

The main motivation was to learn how to create an estimator in sk-learn. Also,
as a main feature compared to the sk-learn KernelDensity class, it implements
boundary conditions for 1D distributions which do not tend to zero in their
boundaries. However, bear in mind this class is slower compared to the sk-learn
implementation. A rough modification to the KernelDensity class (called
KernelDensityMod) has also been done to include boundary conditions.


### Requirements and installation

Packages needed: sklearn, scipy, numpy
To install, simply download the repo and add it to PYTHONPATH

### License

Modified BSD 3 Clause.
