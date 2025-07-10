Scenarios Reference
==================

Scenarios in CausalBoundingEngine organize algorithms by the causal setting and data structure. Each scenario defines which algorithms are applicable and provides a unified interface for accessing them.

Scenario Architecture
---------------------

Base Scenario Class
~~~~~~~~~~~~~~~~~~~

All scenarios inherit from the base ``Scenario`` class, which provides:

- **Data Management**: Handles X, Y, and optional Z variables
- **Algorithm Dispatchers**: ATE and PNS dispatchers for dynamic algorithm access
- **Algorithm Discovery**: Methods to find available algorithms

.. code-block:: python

   from causalboundingengine.scenario import Scenario
   
   class MyScenario(Scenario):
       AVAILABLE_ALGORITHMS = {
           'ATE': {
               'algorithm_name': AlgorithmClass,
           },
           'PNS': {
               'algorithm_name': AlgorithmClass,
           }
       }

Algorithm Dispatchers
~~~~~~~~~~~~~~~~~~~~~

Each scenario provides ATE and PNS dispatchers that expose algorithms as methods:

.. code-block:: python

   scenario = BinaryConf(X, Y)
   
   # These calls are equivalent:
   bounds1 = scenario.ATE.manski()
   bounds2 = scenario.ATE.__getattr__('manski')()
   
   # Dynamic access
   algorithm_name = 'manski'
   bounds3 = getattr(scenario.ATE, algorithm_name)()

Available Scenarios
-------------------

BinaryConf: Binary Confounded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Handle binary treatment and outcome data with potential unmeasured confounding.

**Causal Assumptions**:
   - Binary treatment X ∈ {0, 1}
   - Binary outcome Y ∈ {0, 1}  
   - Unmeasured confounders U may exist
   - No valid instruments available

**Causal Graph**:

.. code-block:: text

   U (unmeasured)
   ↓   ↓
   X → Y

**Data Requirements**:
   - ``X``: NumPy array of 0s and 1s (treatment)
   - ``Y``: NumPy array of 0s and 1s (outcome)
   - Same length for X and Y

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   # Example data
   X = np.array([0, 1, 1, 0, 1, 0, 1])
   Y = np.array([0, 1, 0, 0, 1, 1, 1])
   
   # Create scenario
   scenario = BinaryConf(X, Y)
   
   # Access algorithms
   print("Available ATE algorithms:", scenario.get_algorithms('ATE'))
   print("Available PNS algorithms:", scenario.get_algorithms('PNS'))

**Available Algorithms**:

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Algorithm
     - ATE
     - PNS
     - Notes
   * - manski
     - ✓
     - ✗
     - Most conservative bounds
   * - tianpearl
     - ✓
     - ✓
     - Nonparametric bounds
   * - entropybounds
     - ✓
     - ✓
     - Requires theta parameter
   * - causaloptim
     - ✓
     - ✓
     - Requires R
   * - zaffalonbounds
     - ✓
     - ✓
     - Requires Java
   * - autobound
     - ✓
     - ✓
     - General optimization approach

**Example Usage**:

.. code-block:: python

   # Compute ATE bounds with different algorithms
   manski_bounds = scenario.ATE.manski()
   tianpearl_bounds = scenario.ATE.tianpearl()
   entropy_bounds = scenario.ATE.entropybounds(theta=0.5)
   
   print(f"Manski: {manski_bounds}")
   print(f"Tian-Pearl: {tianpearl_bounds}")
   print(f"Entropy (θ=0.5): {entropy_bounds}")
   
   # Compute PNS bounds
   pns_tianpearl = scenario.PNS.tianpearl()
   pns_entropy = scenario.PNS.entropybounds(theta=0.5)
   
   print(f"PNS Tian-Pearl: {pns_tianpearl}")
   print(f"PNS Entropy: {pns_entropy}")

**When to Use**:
   - Standard observational studies
   - When confounding is suspected
   - No valid instruments available
   - Most common scenario

BinaryIV: Binary Instrumental Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Handle binary treatment, outcome, and instrument data using instrumental variable assumptions.

**Causal Assumptions**:
   - Binary instrument Z ∈ {0, 1}
   - Binary treatment X ∈ {0, 1}
   - Binary outcome Y ∈ {0, 1}
   - Standard IV assumptions:
     
     * **Relevance**: Z affects X
     * **Exclusion**: Z only affects Y through X
     * **Exogeneity**: Z is unconfounded

**Causal Graph**:

.. code-block:: text

   Z → X → Y
       ↑   ↑
        U (unmeasured)

**Data Requirements**:
   - ``Z``: NumPy array of 0s and 1s (instrument)
   - ``X``: NumPy array of 0s and 1s (treatment)
   - ``Y``: NumPy array of 0s and 1s (outcome)
   - All arrays must have the same length

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryIV
   import numpy as np
   
   # Example IV data
   Z = np.array([0, 1, 1, 0, 1, 0, 1])  # Instrument
   X = np.array([0, 1, 1, 0, 1, 0, 0])  # Treatment (influenced by Z)
   Y = np.array([0, 1, 0, 0, 1, 1, 0])  # Outcome
   
   # Create IV scenario
   scenario = BinaryIV(X, Y, Z)

**Available Algorithms**:

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Algorithm
     - ATE
     - PNS
     - Notes
   * - causaloptim
     - ✓
     - ✓
     - Requires R, symbolic approach
   * - zaffalonbounds
     - ✓
     - ✓
     - Requires Java, credal networks
   * - autobound
     - ✓
     - ✓
     - Core Python, LP approach

**Example Usage**:

.. code-block:: python

   # Compute bounds using IV algorithms
   autobound_ate = scenario.ATE.autobound()
   autobound_pns = scenario.PNS.autobound()
   
   print(f"AutoBound ATE: {autobound_ate}")
   print(f"AutoBound PNS: {autobound_pns}")
   
   # If R is available
   try:
       causaloptim_ate = scenario.ATE.causaloptim()
       print(f"CausalOptim ATE: {causaloptim_ate}")
   except ImportError:
       print("R support not available")

**When to Use**:
   - Randomized controlled trials with non-compliance
   - Natural experiments with valid instruments
   - When IV assumptions can be justified
   - Need to account for endogeneity

**IV Validation**:

Before using BinaryIV, validate your instrument:

.. code-block:: python

   import numpy as np
   from scipy.stats import chi2_contingency
   
   def validate_instrument(Z, X, Y):
       \"\"\"Basic IV validation checks.\"\"\"
       # Relevance: Z should be associated with X
       contingency_zx = np.array([[np.sum((Z==0) & (X==0)), np.sum((Z==0) & (X==1))],
                                  [np.sum((Z==1) & (X==0)), np.sum((Z==1) & (X==1))]])
       chi2_zx, p_zx = chi2_contingency(contingency_zx)[:2]
       
       print(f"Relevance test (Z-X association): χ² = {chi2_zx:.3f}, p = {p_zx:.3f}")
       
       # Exclusion is untestable, but we can check Z-Y association conditional on X
       # This shouldn't be strong if exclusion holds
       for x in [0, 1]:
           mask = X == x
           if np.sum(mask) > 10:  # Enough observations
               z_sub = Z[mask]
               y_sub = Y[mask]
               corr = np.corrcoef(z_sub, y_sub)[0, 1]
               print(f"Z-Y correlation given X={x}: {corr:.3f}")
   
   validate_instrument(Z, X, Y)

ContIV: Continuous Instrumental Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Handle continuous variables in instrumental variable settings.

**Causal Assumptions**:
   - Continuous instrument Z
   - Continuous treatment X  
   - Continuous outcome Y
   - Standard IV assumptions hold
   - Variables will be discretized internally

**Data Requirements**:
   - ``Z``: NumPy array of continuous values (instrument)
   - ``X``: NumPy array of continuous values (treatment)
   - ``Y``: NumPy array of continuous values (outcome)
   - All arrays must have the same length

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import ContIV
   import numpy as np
   
   # Generate continuous IV data
   np.random.seed(42)
   n = 200
   
   # Instrument
   Z = np.random.normal(0, 1, n)
   
   # Treatment (influenced by Z and unobserved U)
   U = np.random.normal(0, 1, n)  # Unobserved confounder
   X = 0.5 * Z + 0.3 * U + np.random.normal(0, 0.5, n)
   
   # Outcome (influenced by X and U)
   Y = 0.4 * X + 0.2 * U + np.random.normal(0, 0.5, n)
   
   # Create continuous IV scenario
   scenario = ContIV(X, Y, Z)

**Available Algorithms**:

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Algorithm
     - ATE
     - PNS
     - Notes
   * - zhangbareinboim
     - ✓
     - ✗
     - Handles compliance types

**Example Usage**:

.. code-block:: python

   # Compute ATE bounds for continuous IV
   ate_bounds = scenario.ATE.zhangbareinboim()
   print(f"Zhang-Bareinboim ATE bounds: {ate_bounds}")

**When to Use**:
   - Continuous instrumental variables
   - Economic applications (e.g., distance as instrument)
   - Natural experiments with continuous treatments
   - When discretization is acceptable

**Data Preprocessing**:

ContIV internally discretizes continuous variables. You can also preprocess manually:

.. code-block:: python

   def discretize_variable(var, n_bins=3, method='quantile'):
       \"\"\"Discretize continuous variable.\"\"\"
       if method == 'quantile':
           # Equal-frequency bins
           bin_edges = np.quantile(var, np.linspace(0, 1, n_bins + 1))
       else:
           # Equal-width bins
           bin_edges = np.linspace(var.min(), var.max(), n_bins + 1)
       
       return np.digitize(var, bin_edges) - 1
   
   # Manual discretization
   Z_discrete = discretize_variable(Z, n_bins=2)  # Binary instrument
   X_discrete = discretize_variable(X, n_bins=2)  # Binary treatment  
   Y_discrete = discretize_variable(Y, n_bins=2)  # Binary outcome
   
   # Use with BinaryIV scenario
   scenario_discrete = BinaryIV(X_discrete, Y_discrete, Z_discrete)

Scenario Selection Guide
------------------------

Decision Tree
~~~~~~~~~~~~~

1. **What type of variables do you have?**
   
   - All binary → Continue to step 2
   - Some continuous → Consider ContIV or discretize first

2. **Do you have a valid instrument?**
   
   - Yes, binary instrument → Use BinaryIV
   - Yes, continuous instrument → Use ContIV  
   - No instrument → Use BinaryConf

3. **Can you justify IV assumptions?**
   
   - Relevance: Instrument affects treatment
   - Exclusion: Instrument only affects outcome through treatment
   - Exogeneity: Instrument is unconfounded
   
   If unsure, use BinaryConf for robustness

Scenario Comparison
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Aspect
     - BinaryConf
     - BinaryIV
     - ContIV
     - Notes
   * - Data Type
     - Binary X, Y
     - Binary Z, X, Y
     - Continuous Z, X, Y
     - ContIV discretizes internally
   * - Assumptions
     - Minimal
     - IV assumptions
     - IV assumptions
     - BinaryConf most robust
   * - Algorithms
     - 6 options
     - 3 options
     - 1 option
     - More options = more flexibility
   * - Use Cases
     - Observational
     - RCTs, Natural exp.
     - Economic studies
     - Match study design
   * - Bounds
     - Often wider
     - Can be tighter
     - Varies
     - IV leverages more info

Custom Scenarios
----------------

Creating New Scenarios
~~~~~~~~~~~~~~~~~~~~~~

You can create custom scenarios for specialized use cases:

.. code-block:: python

   from causalboundingengine.scenario import Scenario
   from causalboundingengine.algorithms.manski import Manski
   from causalboundingengine.algorithms.tianpearl import TianPearl
   
   class CustomScenario(Scenario):
       \"\"\"Custom scenario with specific algorithm subset.\"\"\"
       
       AVAILABLE_ALGORITHMS = {
           'ATE': {
               'manski': Manski,
               'tianpearl': TianPearl,
           },
           'PNS': {
               'tianpearl': TianPearl,
           }
       }
       
       def __init__(self, X, Y, additional_data=None):
           super().__init__(X, Y)
           self.additional_data = additional_data
   
   # Use custom scenario
   scenario = CustomScenario(X, Y, additional_data=some_data)
   bounds = scenario.ATE.manski()

Extending Existing Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add algorithms to existing scenarios:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   from causalboundingengine.algorithms.my_algorithm import MyAlgorithm
   
   # Extend BinaryConf
   class ExtendedBinaryConf(BinaryConf):
       AVAILABLE_ALGORITHMS = {
           **BinaryConf.AVAILABLE_ALGORITHMS,
           'ATE': {
               **BinaryConf.AVAILABLE_ALGORITHMS['ATE'],
               'my_algorithm': MyAlgorithm,
           }
       }
   
   scenario = ExtendedBinaryConf(X, Y)
   bounds = scenario.ATE.my_algorithm()

Best Practices
--------------

Data Validation
~~~~~~~~~~~~~~~

Always validate your data before creating scenarios:

.. code-block:: python

   def validate_binary_data(X, Y, Z=None):
       \"\"\"Validate binary data for scenarios.\"\"\"
       arrays = [X, Y] if Z is None else [X, Y, Z]
       names = ['X', 'Y'] if Z is None else ['X', 'Y', 'Z']
       
       for arr, name in zip(arrays, names):
           # Check type
           if not isinstance(arr, np.ndarray):
               raise TypeError(f"{name} must be numpy array")
           
           # Check values
           unique_vals = np.unique(arr)
           if not set(unique_vals).issubset({0, 1}):
               raise ValueError(f"{name} must contain only 0s and 1s, got {unique_vals}")
           
           # Check length
           if len(arr) == 0:
               raise ValueError(f"{name} cannot be empty")
       
       # Check equal lengths
       lengths = [len(arr) for arr in arrays]
       if len(set(lengths)) > 1:
           raise ValueError(f"All arrays must have same length, got {lengths}")
       
       print("Data validation passed")

Scenario Choice Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate that your scenario choice matches your study design:

.. code-block:: python

   def check_scenario_assumptions(scenario_type, Z=None):
       \"\"\"Check if scenario assumptions are met.\"\"\"
       
       if scenario_type == 'BinaryIV':
           if Z is None:
               raise ValueError("BinaryIV requires instrument Z")
           print("Remember to validate IV assumptions:")
           print("- Relevance: Z affects X")
           print("- Exclusion: Z only affects Y through X")  
           print("- Exogeneity: Z is unconfounded")
       
       elif scenario_type == 'BinaryConf':
           if Z is not None:
               print("Warning: Ignoring Z in BinaryConf scenario")
           print("BinaryConf assumes potential confounding")
       
       elif scenario_type == 'ContIV':
           print("ContIV will discretize continuous variables")
           print("Consider manual discretization for control")

Algorithm Selection Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Develop a systematic approach to algorithm selection:

.. code-block:: python

   def select_algorithms(scenario, query='ATE', criteria='all'):
       \"\"\"Select algorithms based on criteria.\"\"\"
       available = scenario.get_algorithms(query)
       
       if criteria == 'fast':
           # Fast algorithms only
           fast_algs = ['manski', 'tianpearl']
           return [alg for alg in available if alg in fast_algs]
       
       elif criteria == 'no_external':
           # No R or Java dependencies
           external_algs = ['causaloptim', 'zaffalonbounds']
           return [alg for alg in available if alg not in external_algs]
       
       elif criteria == 'robust':
           # Multiple different approaches
           robust_set = ['manski', 'tianpearl', 'autobound']
           return [alg for alg in available if alg in robust_set]
       
       else:  # 'all'
           return available
   
   # Usage
   scenario = BinaryConf(X, Y)
   algorithms = select_algorithms(scenario, query='ATE', criteria='robust')
   print(f"Selected algorithms: {algorithms}")

Common Pitfalls
---------------

**Pitfall 1**: Using BinaryIV without validating IV assumptions

.. code-block:: python

   # Bad: Assuming any Z is a valid instrument
   scenario = BinaryIV(X, Y, Z)
   
   # Good: Validate instrument first
   validate_instrument(Z, X, Y)
   if iv_assumptions_hold:
       scenario = BinaryIV(X, Y, Z)
   else:
       scenario = BinaryConf(X, Y)  # Fall back to confounded

**Pitfall 2**: Ignoring algorithm availability

.. code-block:: python

   # Bad: Assuming algorithm is available
   bounds = scenario.ATE.causaloptim()  # May fail if R not installed
   
   # Good: Check availability first
   if 'causaloptim' in scenario.get_algorithms('ATE'):
       try:
           bounds = scenario.ATE.causaloptim()
       except ImportError:
           bounds = scenario.ATE.autobound()  # Fallback

**Pitfall 3**: Not considering data size limitations

.. code-block:: python

   # For very large datasets, some algorithms may be slow
   if len(X) > 10000:
       # Use fast algorithms for exploration
       quick_bounds = scenario.ATE.manski()
   else:
       # Use comprehensive approach
       detailed_bounds = scenario.ATE.autobound()
