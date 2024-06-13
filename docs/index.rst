.. DTAIDistance documentation master file, created by
   sphinx-quickstart on Sun Apr  8 12:55:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DTAIDistance's documentation!
========================================

Library for time series distances (e.g. Dynamic Time Warping) used in
the `DTAI Research Group <https://dtai.cs.kuleuven.be>`__. The library
offers a pure Python implementation and a faster implementation in C.
The C implementation has only Cython as a dependency. It is compatible
with Numpy and Pandas and implemented to avoid unnecessary data copy
operations.

Citing this work: |DOI|

Source available on https://github.com/wannesm/dtaidistance.



.. toctree::
   :caption: Usage
   :maxdepth: 2

   usage/installation
   usage/dtw
   usage/ed
   usage/clustering
   usage/subsequence
   usage/sequence
   usage/similarity
   usage/advanced
   usage/changelist



.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/dtw
   modules/dtw_visualisation
   modules/dtw_ndim
   modules/dtw_barycenter
   modules/ed
   modules/clustering
   modules/subsequence
   modules/preprocessing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |DOI| image:: https://zenodo.org/badge/80764246.svg
   :target: https://zenodo.org/badge/latestdoi/80764246

