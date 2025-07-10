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

Testing Your Algorithm
~~~~~~~~~~~~~~~~~~~~~~

Create tests for your new algorithm:

.. code-block:: python

   # File: tests/test_simple_bounds.py
   
   import numpy as np
   import pytest
   from causalboundingengine.algorithms.simple_bounds import SimpleBounds
   
   class TestSimpleBounds:
       
       def test_ate_basic(self):
           \"\"\"Test basic ATE computation.\"\"\"
           X = np.array([0, 1, 1, 0])
           Y = np.array([0, 1, 1, 0])
           
           alg = SimpleBounds()
           lower, upper = alg.bound_ATE(X, Y)
           
           assert isinstance(lower, float)
           assert isinstance(upper, float)
           assert lower <= upper
           assert -1 <= lower <= 1
           assert -1 <= upper <= 1
       
       def test_pns_basic(self):
           \"\"\"Test basic PNS computation.\"\"\"
           X = np.array([0, 1, 1, 0])
           Y = np.array([0, 1, 1, 0])
           
           alg = SimpleBounds()
           lower, upper = alg.bound_PNS(X, Y)
           
           assert isinstance(lower, float)
           assert isinstance(upper, float)
           assert lower <= upper
           assert 0 <= lower <= 1
           assert 0 <= upper <= 1
       
       def test_edge_cases(self):
           \"\"\"Test edge cases.\"\"\"
           # All treated
           X = np.array([1, 1, 1, 1])
           Y = np.array([0, 1, 1, 0])
           
           alg = SimpleBounds()
           lower, upper = alg.bound_ATE(X, Y)
           assert not np.isnan(lower)
           assert not np.isnan(upper)

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
       
       def __init__(self, X, Y, additional_data=None):
           \"\"\"Initialize custom scenario.
           
           Args:
               X: Treatment array
               Y: Outcome array  
               additional_data: Custom data for this scenario
           \"\"\"
           super().__init__(X, Y)
           self.additional_data = additional_data
       
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

Advanced Algorithm Features
---------------------------

Caching Results
~~~~~~~~~~~~~~

For expensive computations, implement caching:

.. code-block:: python

   import functools
   import hashlib
   import pickle
   
   class CachedAlgorithm(Algorithm):
       \"\"\"Algorithm with result caching.\"\"\"
       
       @functools.lru_cache(maxsize=128)
       def _cached_compute(self, data_hash: str, **kwargs) -> tuple[float, float]:
           \"\"\"Cached computation method.\"\"\"
           # Expensive computation here
           return lower_bound, upper_bound
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> tuple[float, float]:
           # Create hash of input data
           data_bytes = pickle.dumps((X.tolist(), Y.tolist(), sorted(kwargs.items())))
           data_hash = hashlib.md5(data_bytes).hexdigest()
           
           return self._cached_compute(data_hash, **kwargs)

Progress Tracking
~~~~~~~~~~~~~~~~

For long-running algorithms, add progress tracking:

.. code-block:: python

   from tqdm import tqdm
   import time
   
   class ProgressAlgorithm(Algorithm):
       \"\"\"Algorithm with progress tracking.\"\"\"
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, 
                       show_progress: bool = True, **kwargs) -> tuple[float, float]:
           \"\"\"Compute with progress bar.\"\"\"
           
           n_iterations = 100  # Number of computation steps
           
           if show_progress:
               pbar = tqdm(total=n_iterations, desc=\"Computing bounds\")
           
           result = 0.0
           for i in range(n_iterations):
               # Simulation step
               time.sleep(0.01)  # Simulate work
               result += np.random.normal(0, 0.01)
               
               if show_progress:
                   pbar.update(1)
           
           if show_progress:
               pbar.close()
           
           return result - 0.1, result + 0.1

Parallel Computing
~~~~~~~~~~~~~~~~~

For algorithms that can benefit from parallelization:

.. code-block:: python

   from multiprocessing import Pool
   import numpy as np
   
   class ParallelAlgorithm(Algorithm):
       \"\"\"Algorithm using parallel computation.\"\"\"
       
       def _bootstrap_iteration(self, args):
           \"\"\"Single bootstrap iteration.\"\"\"
           X, Y, seed = args
           np.random.seed(seed)
           
           # Bootstrap sample
           n = len(X)
           indices = np.random.choice(n, n, replace=True)
           X_boot = X[indices]
           Y_boot = Y[indices]
           
           # Compute bounds for this sample
           p1 = np.mean(Y_boot[X_boot == 1]) if np.any(X_boot == 1) else 0.0
           p0 = np.mean(Y_boot[X_boot == 0]) if np.any(X_boot == 0) else 0.0
           
           return p1 - p0
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, 
                       n_bootstrap: int = 100, n_jobs: int = -1, **kwargs) -> tuple[float, float]:
           \"\"\"Parallel bootstrap computation.\"\"\"
           
           if n_jobs == -1:
               n_jobs = None  # Use all CPUs
           
           # Prepare arguments for parallel processing
           args = [(X, Y, i) for i in range(n_bootstrap)]
           
           # Parallel computation
           with Pool(n_jobs) as pool:
               results = pool.map(self._bootstrap_iteration, args)
           
           # Compute bounds from bootstrap results
           lower_bound = np.percentile(results, 2.5)
           upper_bound = np.percentile(results, 97.5)
           
           return lower_bound, upper_bound

Documentation Standards
-----------------------

Docstring Format
~~~~~~~~~~~~~~~

Use NumPy-style docstrings for consistency:

.. code-block:: python

   class WellDocumentedAlgorithm(Algorithm):
       \"\"\"
       Well-documented algorithm for demonstration.
       
       This algorithm demonstrates proper documentation standards
       for CausalBoundingEngine algorithms.
       
       References
       ----------
       Author, A. (2023). Important Paper. Journal of Causal Inference.
       
       Examples
       --------
       >>> import numpy as np
       >>> from causalboundingengine.scenarios import BinaryConf
       >>> X = np.array([0, 1, 1, 0])
       >>> Y = np.array([1, 0, 1, 1])
       >>> scenario = BinaryConf(X, Y)
       >>> bounds = scenario.ATE.well_documented()
       >>> print(bounds)
       (0.1, 0.9)
       \"\"\"
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, 
                       parameter: float = 1.0, **kwargs) -> tuple[float, float]:
           \"\"\"
           Compute ATE bounds using the well-documented method.
           
           Parameters
           ----------
           X : np.ndarray
               Binary treatment array of shape (n,) with values in {0, 1}.
           Y : np.ndarray  
               Binary outcome array of shape (n,) with values in {0, 1}.
           parameter : float, default=1.0
               Algorithm-specific parameter. Must be positive.
           **kwargs
               Additional keyword arguments (ignored).
               
           Returns
           -------
           tuple[float, float]
               Lower and upper bounds on the ATE.
               
           Raises
           ------
           ValueError
               If parameter is not positive.
               
           Notes
           -----
           This method implements the algorithm described in Author (2023).
           The bounds are computed by...
           
           The computational complexity is O(n) where n is the sample size.
           \"\"\"
           
           if parameter <= 0:
               raise ValueError(\"parameter must be positive\")
           
           # Implementation here
           return 0.0, 1.0

README Documentation
~~~~~~~~~~~~~~~~~~~

If creating a new algorithm file, include a header comment:

.. code-block:: python

   \"\"\"
   Well-Documented Algorithm
   ========================
   
   This module implements the Well-Documented algorithm for causal bounding.
   
   The algorithm is based on the work of Author et al. (2023) and provides
   bounds on causal effects under specific assumptions.
   
   Key Features:
   - Fast computation (O(n) complexity)
   - Handles missing data gracefully
   - Configurable sensitivity parameter
   
   Dependencies:
   - numpy
   - scipy (optional, for advanced features)
   
   Example Usage:
   -------------
   >>> from causalboundingengine.algorithms.well_documented import WellDocumentedAlgorithm
   >>> alg = WellDocumentedAlgorithm()
   >>> bounds = alg.bound_ATE(X, Y, parameter=0.5)
   
   References:
   ----------
   Author, A., Coauthor, B. (2023). \"Important Method for Causal Bounds.\"
   Journal of Causal Inference, 15(2), 123-145.
   \"\"\"

Contributing to CausalBoundingEngine
------------------------------------

Development Setup
~~~~~~~~~~~~~~~~

1. **Fork and Clone**:

.. code-block:: bash

   git clone https://github.com/yourusername/CausalBoundingEngine.git
   cd CausalBoundingEngine

2. **Install Development Dependencies**:

.. code-block:: bash

   pip install -e .[full,docs]
   pip install pytest pytest-cov black isort mypy

3. **Create Feature Branch**:

.. code-block:: bash

   git checkout -b feature/my-new-algorithm

Code Quality Standards
~~~~~~~~~~~~~~~~~~~~~

**Formatting**: Use Black for code formatting:

.. code-block:: bash

   black causalboundingengine/ tests/

**Import Sorting**: Use isort:

.. code-block:: bash

   isort causalboundingengine/ tests/

**Type Checking**: Use mypy:

.. code-block:: bash

   mypy causalboundingengine/

**Testing**: Ensure good test coverage:

.. code-block:: bash

   pytest tests/ --cov=causalboundingengine

Pull Request Process
~~~~~~~~~~~~~~~~~~~

1. **Add Tests**: Include comprehensive tests for new functionality
2. **Update Documentation**: Add docstrings and update relevant .rst files
3. **Run Quality Checks**: Ensure all code quality tools pass
4. **Update CHANGELOG**: Document your changes
5. **Submit PR**: Provide clear description of changes

Example Pull Request Checklist:

.. code-block:: text

   - [ ] Added comprehensive tests
   - [ ] Updated documentation
   - [ ] Ran black, isort, mypy
   - [ ] All tests pass
   - [ ] Added example usage
   - [ ] Updated relevant scenario files
   - [ ] Added algorithm to __init__.py if needed

File Organization
~~~~~~~~~~~~~~~~

Place new files in the appropriate locations:

.. code-block:: text

   causalboundingengine/
   ├── algorithms/
   │   ├── __init__.py
   │   ├── algorithm.py          # Base class
   │   ├── my_algorithm.py       # Your algorithm
   │   └── ...
   ├── scenarios.py              # Update with new algorithm
   └── ...
   
   tests/
   ├── test_my_algorithm.py      # Your tests
   └── ...
   
   docs/source/
   ├── algorithms.rst            # Update algorithm docs
   └── ...

Integration Examples
-------------------

Complete Example: Adding a New Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example of adding a new algorithm:

**1. Algorithm Implementation** (``causalboundingengine/algorithms/bayesian_bounds.py``):

.. code-block:: python

   \"\"\"
   Bayesian Bounds Algorithm
   ========================
   
   Implements Bayesian bounds using prior information.
   \"\"\"
   
   import numpy as np
   from scipy import stats
   from causalboundingengine.algorithms.algorithm import Algorithm
   
   class BayesianBounds(Algorithm):
       \"\"\"Bayesian bounds using prior distributions.\"\"\"
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, 
                       prior_mean: float = 0.0, prior_std: float = 0.5,
                       **kwargs) -> tuple[float, float]:
           \"\"\"Compute ATE bounds using Bayesian approach.\"\"\"
           
           # Observed data
           n1 = np.sum(X == 1)
           n0 = np.sum(X == 0)
           y1_sum = np.sum(Y[X == 1]) if n1 > 0 else 0
           y0_sum = np.sum(Y[X == 0]) if n0 > 0 else 0
           
           # Bayesian updates (Beta-Binomial conjugate)
           # Prior: Beta(1, 1) (uniform)
           # Posterior: Beta(1 + successes, 1 + failures)
           
           if n1 > 0:
               p1_posterior = stats.beta(1 + y1_sum, 1 + n1 - y1_sum)
               p1_lower, p1_upper = p1_posterior.interval(0.95)
           else:
               p1_lower, p1_upper = 0.0, 1.0
           
           if n0 > 0:
               p0_posterior = stats.beta(1 + y0_sum, 1 + n0 - y0_sum)
               p0_lower, p0_upper = p0_posterior.interval(0.95)
           else:
               p0_lower, p0_upper = 0.0, 1.0
           
           # ATE bounds
           ate_lower = p1_lower - p0_upper
           ate_upper = p1_upper - p0_lower
           
           return float(ate_lower), float(ate_upper)

**2. Add Tests** (``tests/test_bayesian_bounds.py``):

.. code-block:: python

   import numpy as np
   import pytest
   from causalboundingengine.algorithms.bayesian_bounds import BayesianBounds
   
   class TestBayesianBounds:
       
       def test_basic_computation(self):
           X = np.array([0, 1, 1, 0, 1])
           Y = np.array([0, 1, 0, 0, 1])
           
           alg = BayesianBounds()
           lower, upper = alg.bound_ATE(X, Y)
           
           assert isinstance(lower, float)
           assert isinstance(upper, float)
           assert lower <= upper

**3. Update Scenarios** (in ``causalboundingengine/scenarios.py``):

.. code-block:: python

   # Add import
   from causalboundingengine.algorithms.bayesian_bounds import BayesianBounds
   
   # Update BinaryConf class
   class BinaryConf(Scenario):
       AVAILABLE_ALGORITHMS = {
           'ATE': {
               'manski': Manski,
               'tianpearl': TianPearl,
               # ... existing algorithms ...
               'bayesian_bounds': BayesianBounds,  # Add here
           },
           # ...
       }

**4. Update Documentation** (in ``docs/source/algorithms.rst``):

Add a section describing your new algorithm, its parameters, usage, and when to use it.

**5. Test Integration**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   X = np.array([0, 1, 1, 0])
   Y = np.array([1, 0, 1, 1])
   scenario = BinaryConf(X, Y)
   
   # Should now work
   bounds = scenario.ATE.bayesian_bounds()
   print(f\"Bayesian bounds: {bounds}\")

This complete example shows the full workflow for adding a new algorithm to CausalBoundingEngine.
