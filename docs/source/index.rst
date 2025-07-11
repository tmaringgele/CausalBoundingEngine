CausalBoundingEngine Documentation
===================================

.. image:: _static/cbe_logo.png
   :align: center
   :alt: CausalBoundingEngine Logo
   :width: 400px

|

CausalBoundingEngine is a modular Python package for comparing and applying causal bounding algorithms. It provides a unified interface for computing bounds on causal effects in scenarios with unmeasured confounding.

.. image:: https://img.shields.io/pypi/v/causalboundingengine.svg
   :target: https://pypi.org/project/causalboundingengine/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/yourusername/CausalBoundingEngine/blob/main/LICENSE
   :alt: MIT License

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install causalboundingengine

Basic usage:

.. code-block:: python

   import numpy as np
   from causalboundingengine.scenarios import BinaryConf

   # Your data
   X = np.array([0, 1, 1, 0, 1])  # Treatment
   Y = np.array([1, 0, 1, 0, 1])  # Outcome

   # Create scenario and compute bounds
   scenario = BinaryConf(X, Y)
   ate_bounds = scenario.ATE.manski()  # (-1.0, 1.0)
   pns_bounds = scenario.PNS.tianpearl()  # (0.0, 0.8)

Features
--------

- **Multiple Algorithms**: Manski, Tian-Pearl, Causaloptim, Autobound, ZhangBareinboim, Zaffalonbounds, and more
- **Unified Interface**: Consistent API across all algorithms and scenarios
- **Multiple Scenarios**: Support for confounded and instrumental variable settings
- **External Engine Support**: Integration with R (via rpy2) and Java (via jpype1)
- **Extensible Design**: Easy to add new algorithms and scenarios

.. note::
   **Attribution**: CausalBoundingEngine integrates algorithms from multiple published research papers. Please see the :doc:`references` section for complete citations and attribution when using this library in your research.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   quickstart
   core_concepts
   algorithms
   scenarios
   extending
   api_reference
   examples
   references
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

