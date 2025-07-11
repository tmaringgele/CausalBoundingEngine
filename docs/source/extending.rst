Extending CausalBoundingEngine
==============================

CausalBoundingEngine is designed to be easily extensible. This guide shows how to add new algorithms, create custom scenarios, and contribute to the project.

Adding New Algorithms
---------------------

Algorithm Structure
~~~~~~~~~~~~~~~~~~~

All algorithms must inherit from the base ``Algorithm`` class and implement the required methods:

.. code-block:: python

   from causalboundingengine.algorithms.algorithm import Algorithm
   import numpy as np
   
   class MyAlgorithm(Algorithm):
       \"\"\"Template for a new algorithm.\"\"\"
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray = None, **kwargs) -> tuple[float, float]:
           \"\"\"Compute ATE bounds.
           
           Args:
               X: Binary treatment array (0s and 1s)
               Y: Binary outcome array (0s and 1s) 
               Z: Optional binary instrument array (0s and 1s)
               **kwargs: Additional algorithm-specific parameters
               
           Returns:
               tuple: (lower_bound, upper_bound)
           \"\"\"
           # Your algorithm implementation here
           lower_bound = -1.0  # Replace with actual computation
           upper_bound = 1.0   # Replace with actual computation
           return lower_bound, upper_bound
       
       def _compute_PNS(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray = None, **kwargs) -> tuple[float, float]:
           \"\"\"Compute PNS bounds.
           
           Args:
               X: Binary treatment array
               Y: Binary outcome array
               Z: Optional binary instrument array
               **kwargs: Additional parameters
               
           Returns:
               tuple: (lower_bound, upper_bound)
           \"\"\"
           # Your PNS implementation here
           lower_bound = 0.0   # Replace with actual computation
           upper_bound = 1.0   # Replace with actual computation
           return lower_bound, upper_bound

Key Requirements
~~~~~~~~~~~~~~~~

1. **Inherit from Algorithm**: Your class must extend ``Algorithm``
2. **Implement _compute_* methods**: At least one of ``_compute_ATE`` or ``_compute_PNS``
3. **Return tuple**: Always return ``(lower_bound, upper_bound)``
4. **Handle errors gracefully**: The base class will catch exceptions and return trivial bounds
5. **Type hints**: Use proper type annotations for clarity

Example: Simple New Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's implement a simple algorithm that computes bounds based on observed proportions:

.. code-block:: python

   # File: causalboundingengine/algorithms/simple_bounds.py
   
   import numpy as np
   from causalboundingengine.algorithms.algorithm import Algorithm
   
   class SimpleBounds(Algorithm):
       \"\"\"Simple bounds based on observed proportions.\"\"\"
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> tuple[float, float]:
           \"\"\"Compute ATE using simple proportion-based bounds.\"\"\"
           
           # Observed proportions
           p1 = np.mean(Y[X == 1]) if np.any(X == 1) else 0.0  # P(Y=1|X=1)
           p0 = np.mean(Y[X == 0]) if np.any(X == 0) else 0.0  # P(Y=1|X=0)
           
           # Simple bounds: assume worst case for unobserved
           ate_observed = p1 - p0
           margin = 0.5  # Conservative margin
           
           lower_bound = ate_observed - margin
           upper_bound = ate_observed + margin
           
           # Ensure bounds are within [-1, 1]
           lower_bound = max(lower_bound, -1.0)
           upper_bound = min(upper_bound, 1.0)
           
           return lower_bound, upper_bound
       
       def _compute_PNS(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> tuple[float, float]:
           \"\"\"Compute PNS using observed joint probabilities.\"\"\"
           
           # Observed joint probabilities
           p_11 = np.mean((X == 1) & (Y == 1))  # P(X=1, Y=1)
           p_00 = np.mean((X == 0) & (Y == 0))  # P(X=0, Y=0)
           
           # Conservative PNS bounds
           lower_bound = max(0.0, p_11 + p_00 - 1.0)
           upper_bound = min(p_11, p_00)
           
           return lower_bound, upper_bound

Adding Parameters
~~~~~~~~~~~~~~~~~

Algorithms can accept additional parameters:

.. code-block:: python

   class ParametrizedAlgorithm(Algorithm):
       \"\"\"Algorithm with user-configurable parameters.\"\"\"
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, 
                       sensitivity: float = 0.1, 
                       method: str = 'conservative',
                       **kwargs) -> tuple[float, float]:
           \"\"\"
           Args:
               sensitivity: Sensitivity parameter (0-1)
               method: Method to use ('conservative' or 'optimistic')
           \"\"\"
           
           if not 0 <= sensitivity <= 1:
               raise ValueError(\"sensitivity must be between 0 and 1\")
           
           if method not in ['conservative', 'optimistic']:
               raise ValueError(\"method must be 'conservative' or 'optimistic'\")
           
           # Use parameters in computation
           p1 = np.mean(Y[X == 1]) if np.any(X == 1) else 0.0
           p0 = np.mean(Y[X == 0]) if np.any(X == 0) else 0.0
           
           base_effect = p1 - p0
           
           if method == 'conservative':
               margin = sensitivity
           else:  # optimistic
               margin = sensitivity / 2
           
           return base_effect - margin, base_effect + margin

External Dependencies
~~~~~~~~~~~~~~~~~~~~

For algorithms requiring external libraries:

.. code-block:: python

   class ExternalAlgorithm(Algorithm):
       \"\"\"Algorithm requiring external dependencies.\"\"\"
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> tuple[float, float]:
           try:
               import external_library
           except ImportError:
               raise ImportError(
                   \"external_library is required for ExternalAlgorithm. \"
                   \"Install with: pip install external_library\"
               )
           
           # Use external library
           result = external_library.compute_bounds(X, Y)
           return result.lower, result.upper



Creating Custom Scenarios
--------------------------

Scenario Structure
~~~~~~~~~~~~~~~~~

Scenarios organize algorithms by data structure and causal assumptions:

.. code-block:: python

   from causalboundingengine.scenario import Scenario
   from causalboundingengine.algorithms.simple_bounds import SimpleBounds
   from causalboundingengine.algorithms.manski import Manski
   
   class CustomScenario(Scenario):
       \"\"\"Custom scenario for specific use case.\"\"\"
       
       AVAILABLE_ALGORITHMS = {
           'ATE': {
               'simple_bounds': SimpleBounds,
               'manski': Manski,
           },
           'PNS': {
               'simple_bounds': SimpleBounds,
           }
       }
       
       def custom_method(self):
           \"\"\"Add custom functionality.\"\"\"
           return f\"Custom scenario with {len(self.data.X)} observations\"

Using Custom Scenarios
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use your custom scenario
   import numpy as np
   
   X = np.array([0, 1, 1, 0])
   Y = np.array([1, 0, 1, 1])
   
   scenario = CustomScenario(X, Y, additional_data=\"metadata\")
   
   # Access algorithms
   bounds = scenario.ATE.simple_bounds()
   print(f\"Custom bounds: {bounds}\")
   
   # Use custom methods
   info = scenario.custom_method()
   print(info)

Extending Existing Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add algorithms to existing scenarios without modifying the core code:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   from causalboundingengine.algorithms.simple_bounds import SimpleBounds
   
   class ExtendedBinaryConf(BinaryConf):
       \"\"\"BinaryConf with additional algorithms.\"\"\"
       
       AVAILABLE_ALGORITHMS = {
           # Copy existing algorithms
           'ATE': {
               **BinaryConf.AVAILABLE_ALGORITHMS['ATE'],
               'simple_bounds': SimpleBounds,  # Add new algorithm
           },
           'PNS': {
               **BinaryConf.AVAILABLE_ALGORITHMS['PNS'],
               'simple_bounds': SimpleBounds,
           }
       }
   
   # Use extended scenario
   scenario = ExtendedBinaryConf(X, Y)
   bounds = scenario.ATE.simple_bounds()
