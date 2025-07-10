API Reference
=============

This page provides the complete API documentation for CausalBoundingEngine.

Core Classes
------------

Scenario
~~~~~~~~

.. autoclass:: causalboundingengine.scenario.Scenario
   :members:
   :undoc-members:
   :show-inheritance:

AlgorithmDispatcher
~~~~~~~~~~~~~~~~~~~

.. autoclass:: causalboundingengine.scenario.AlgorithmDispatcher
   :members:
   :undoc-members:
   :show-inheritance:

Data
~~~~

.. autoclass:: causalboundingengine.utils.data.Data
   :members:
   :undoc-members:
   :show-inheritance:

Algorithm Base Class
--------------------

Algorithm
~~~~~~~~~

.. autoclass:: causalboundingengine.algorithms.algorithm.Algorithm
   :members:
   :undoc-members:
   :show-inheritance:

Scenarios
---------

BinaryConf
~~~~~~~~~~

.. autoclass:: causalboundingengine.scenarios.BinaryConf
   :members:
   :undoc-members:
   :show-inheritance:

BinaryIV
~~~~~~~~

.. autoclass:: causalboundingengine.scenarios.BinaryIV
   :members:
   :undoc-members:
   :show-inheritance:

ContIV
~~~~~~

.. autoclass:: causalboundingengine.scenarios.ContIV
   :members:
   :undoc-members:
   :show-inheritance:

Core Algorithms
---------------

Manski
~~~~~~

.. autoclass:: causalboundingengine.algorithms.manski.Manski
   :members:
   :undoc-members:
   :show-inheritance:

TianPearl
~~~~~~~~~

.. autoclass:: causalboundingengine.algorithms.tianpearl.TianPearl
   :members:
   :undoc-members:
   :show-inheritance:

AutoBound
~~~~~~~~~

.. autoclass:: causalboundingengine.algorithms.autobound.Autobound
   :members:
   :undoc-members:
   :show-inheritance:

EntropyBounds
~~~~~~~~~~~~~

.. autoclass:: causalboundingengine.algorithms.entropybounds.Entropybounds
   :members:
   :undoc-members:
   :show-inheritance:

ZhangBareinboim
~~~~~~~~~~~~~~~

.. autoclass:: causalboundingengine.algorithms.zhangbareinboim.ZhangBareinboim
   :members:
   :undoc-members:
   :show-inheritance:

External Engine Algorithms
---------------------------

CausalOptim
~~~~~~~~~~~

.. autoclass:: causalboundingengine.algorithms.causaloptim.CausalOptim
   :members:
   :undoc-members:
   :show-inheritance:

Zaffalonbounds
~~~~~~~~~~~~~~

.. autoclass:: causalboundingengine.algorithms.zaffalonbounds.Zaffalonbounds
   :members:
   :undoc-members:
   :show-inheritance:

Utility Classes
---------------

AlgUtil
~~~~~~~

.. autoclass:: causalboundingengine.utils.alg_util.AlgUtil
   :members:
   :undoc-members:
   :show-inheritance:

Constants and Types
-------------------

Common Types
~~~~~~~~~~~~

The following types are commonly used throughout the API:

.. code-block:: python

   from typing import Tuple
   import numpy as np
   
   # Type aliases used in CausalBoundingEngine
   BoundsResult = Tuple[float, float]
   BinaryArray = np.ndarray  # Array of 0s and 1s
   ContinuousArray = np.ndarray  # Array of continuous values

Common Parameters
~~~~~~~~~~~~~~~~~

Many algorithms accept these standard parameters:

- **X** (``np.ndarray``): Binary treatment array with values in {0, 1}
- **Y** (``np.ndarray``): Binary outcome array with values in {0, 1} 
- **Z** (``np.ndarray``, optional): Binary instrument array with values in {0, 1}

Algorithm-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**EntropyBounds**:
   - ``theta`` (``float``): Information constraint level for mutual information

**CausalOptim**:
   - ``r_path`` (``str``, optional): Custom path to R executable

Return Values
~~~~~~~~~~~~~

All bound computation methods return:

- ``Tuple[float, float]``: ``(lower_bound, upper_bound)`` where ``lower_bound <= upper_bound``

For ATE bounds: ``-1.0 <= lower_bound <= upper_bound <= 1.0``

For PNS bounds: ``0.0 <= lower_bound <= upper_bound <= 1.0``

Exception Handling
------------------

Algorithm Failures
~~~~~~~~~~~~~~~~~~~

When algorithms fail, they return trivial bounds instead of raising exceptions:

- **ATE**: ``(-1.0, 1.0)`` 
- **PNS**: ``(0.0, 1.0)``

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

Algorithms with missing dependencies raise ``ImportError``:

.. code-block:: python

   try:
       bounds = scenario.ATE.causaloptim()
   except ImportError as e:
       print(f"R support not available: {e}")

Invalid Parameters
~~~~~~~~~~~~~~~~~~

Invalid parameters raise ``ValueError`` or ``TypeError``:

.. code-block:: python

   try:
       bounds = scenario.ATE.entropybounds(theta=-1.0)  # Invalid theta
   except ValueError as e:
       print(f"Invalid parameter: {e}")

Usage Patterns
--------------

Basic Pattern
~~~~~~~~~~~~~

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   # Data preparation
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   
   # Scenario creation
   scenario = BinaryConf(X, Y)
   
   # Bound computation
   bounds = scenario.ATE.manski()
   print(f"Bounds: {bounds}")

Dynamic Algorithm Access
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get available algorithms
   algorithms = scenario.get_algorithms('ATE')
   
   # Use algorithms dynamically
   for alg_name in algorithms:
       try:
           alg_func = getattr(scenario.ATE, alg_name)
           bounds = alg_func()
           print(f"{alg_name}: {bounds}")
       except Exception as e:
           print(f"{alg_name} failed: {e}")

Error Handling Pattern
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   
   # Enable warnings for algorithm failures
   logging.basicConfig(level=logging.WARNING)
   
   # Robust computation with fallback
   def compute_bounds_robust(scenario, preferred_alg='autobound', fallback_alg='manski'):
       try:
           return getattr(scenario.ATE, preferred_alg)()
       except Exception:
           return getattr(scenario.ATE, fallback_alg)()
   
   bounds = compute_bounds_robust(scenario)

Extending the API
-----------------

Custom Algorithms
~~~~~~~~~~~~~~~~~

Implement the ``Algorithm`` interface:

.. code-block:: python

   from causalboundingengine.algorithms.algorithm import Algorithm
   
   class MyAlgorithm(Algorithm):
       def _compute_ATE(self, X, Y, **kwargs):
           # Your implementation
           return lower_bound, upper_bound
       
       def _compute_PNS(self, X, Y, **kwargs):
           # Your implementation  
           return lower_bound, upper_bound

Custom Scenarios
~~~~~~~~~~~~~~~~

Extend the ``Scenario`` class:

.. code-block:: python

   from causalboundingengine.scenario import Scenario
   
   class MyScenario(Scenario):
       AVAILABLE_ALGORITHMS = {
           'ATE': {'my_algorithm': MyAlgorithm},
           'PNS': {'my_algorithm': MyAlgorithm},
       }

Version Information
-------------------

Access version information:

.. code-block:: python

   import causalboundingengine
   print(causalboundingengine.__version__)  # Package version

Module Structure
----------------

The package is organized as follows:

.. code-block:: text

   causalboundingengine/
   ├── __init__.py              # Package initialization
   ├── scenario.py              # Base scenario and dispatcher classes
   ├── scenarios.py             # Concrete scenario implementations
   ├── algorithms/              # Algorithm implementations
   │   ├── __init__.py
   │   ├── algorithm.py         # Base algorithm class
   │   ├── manski.py           # Manski bounds
   │   ├── tianpearl.py        # Tian-Pearl bounds
   │   ├── autobound.py        # AutoBound algorithm
   │   ├── entropybounds.py    # Entropy-based bounds
   │   ├── causaloptim.py      # R-based CausalOptim
   │   ├── zaffalonbounds.py   # Java-based Zaffalonbounds
   │   └── zhangbareinboim.py  # Zhang-Bareinboim bounds
   └── utils/                   # Utility functions
       ├── __init__.py
       ├── data.py             # Data handling
       ├── alg_util.py         # Algorithm utilities
       └── r_utils.py          # R integration utilities
