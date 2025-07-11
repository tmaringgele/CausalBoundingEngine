Scenarios Reference
==================

Scenarios in CausalBoundingEngine organize algorithms by the causal setting and data structure. Each scenario defines which algorithms are applicable and provides a unified interface for accessing them.

.. note::
   All scenarios use algorithms from published research. For complete citations and references, see the :doc:`references` page.

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
   
   print(f"Autobound ATE: {autobound_ate}")
   print(f"Autobound PNS: {autobound_pns}")
   
   # If R is available
   try:
       causaloptim_ate = scenario.ATE.causaloptim()
       print(f"Causaloptim ATE: {causaloptim_ate}")
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

ContIV: Binary IV with Continuous Outcome
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Handle binary instrument and treatment with continuous outcome using instrumental variable assumptions.

**Causal Assumptions**:
   - Binary instrument Z ∈ {0, 1}
   - Binary treatment X ∈ {0, 1}  
   - Continuous outcome Y ∈ [0, 1] (bounded between 0 and 1)
   - Standard IV assumptions hold

**Data Requirements**:
   - ``Z``: NumPy array of 0s and 1s (binary instrument)
   - ``X``: NumPy array of 0s and 1s (binary treatment)
   - ``Y``: NumPy array of continuous values between 0 and 1 (outcome)
   - All arrays must have the same length

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import ContIV
   import numpy as np
   
   # Example IV data with continuous outcome
   Z = np.array([0, 1, 1, 0, 1])  # Binary instrument
   X = np.array([0, 1, 1, 0, 1])  # Binary treatment
   Y = np.array([0.1, 0.8, 0.2, 0.9, 0.3])  # Continuous outcome (0-1)
   
   # Create scenario
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

   # Compute ATE bounds for binary IV with continuous outcome
   ate_bounds = scenario.ATE.zhangbareinboim()
   print(f"Zhang-Bareinboim ATE bounds: {ate_bounds}")

**When to Use**:
   - Binary instrumental variables with continuous outcomes
   - RCTs with binary treatment and continuous response measures
   - Economic applications with binary policy instruments
   - When outcome is naturally continuous but bounded

**Important Notes**:
   - Z and X should be binary (0s and 1s only)
   - Y should be continuous values between 0 and 1
   - Algorithm may still run with non-binary Z/X but this is not the intended use
   - For fully continuous variables, consider discretization first

**Data Preprocessing**:

Ensure your outcome is properly bounded:

.. code-block:: python

   def prepare_continuous_outcome(Y, method='min_max'):
       \"\"\"Prepare continuous outcome for ContIV.\"\"\"
       if method == 'min_max':
           # Min-max normalization to [0, 1]
           Y_norm = (Y - Y.min()) / (Y.max() - Y.min())
       elif method == 'sigmoid':
           # Sigmoid transformation
           Y_norm = 1 / (1 + np.exp(-Y))
       else:
           raise ValueError("method must be 'min_max' or 'sigmoid'")
       
       return Y_norm
   
   # Example usage
   Y_raw = np.array([2.1, 5.8, 1.2, 7.9, 3.3])  # Raw continuous outcome
   Y_bounded = prepare_continuous_outcome(Y_raw, method='min_max')
   
   # Use with ContIV
   scenario = ContIV(X, Y_bounded, Z)

Scenario Selection Guide
------------------------

Decision Tree
~~~~~~~~~~~~~

1. **What type of variables do you have?**
   
   - All binary → Continue to step 2
   - Some continuous → Consider ContIV or discretize first

2. **Do you have a valid instrument?**
   
   - Yes, binary instrument, binary outcome → Use BinaryIV
   - Yes, binary instrument, continuous outcome → Use ContIV  
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
     - Binary Z, X; Continuous Y [0,1]
     - ContIV for bounded outcomes
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
