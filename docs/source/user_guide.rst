User Guide
==========

This guide provides a comprehensive overview of CausalBoundingEngine's concepts, scenarios, and usage patterns.

Core Concepts
-------------

Causal Bounding
~~~~~~~~~~~~~~~

Causal inference often faces the challenge of unmeasured confounding - variables that affect both treatment and outcome but are not observed. When identification of causal effects is impossible, **causal bounding** provides a principled approach to determine the range of possible causal effects compatible with the observed data and assumptions.

CausalBoundingEngine focuses on two key causal quantities:

**Average Treatment Effect (ATE)**:
   The difference in expected outcomes between treated and untreated states:
   
   .. math:: ATE = E[Y(1)] - E[Y(0)]

**Probability of Necessity and Sufficiency (PNS)**:
   The probability that treatment is both necessary and sufficient for a positive outcome:
   
   .. math:: PNS = P(Y(1)=1, Y(0)=0)

Data Requirements
~~~~~~~~~~~~~~~~~

CausalBoundingEngine works with binary variables:

- **X**: Binary treatment variable (0 = control, 1 = treated)
- **Y**: Binary outcome variable (0 = failure, 1 = success)  
- **Z**: Optional binary instrument (for IV scenarios)

All data should be provided as NumPy arrays of integers (0s and 1s).

Scenarios
---------

CausalBoundingEngine organizes algorithms by **scenarios** - different causal settings that determine which algorithms are applicable.

BinaryConf: Binary Confounded Setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**: When you suspect unmeasured confounding between treatment and outcome.

**Assumptions**:
   - Binary treatment and outcome
   - Unmeasured confounders may exist
   - No instruments available

**Causal graph**:

.. code-block:: text

   U (unmeasured)
   ↓   ↓
   X → Y

**Example**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   # Observational data with potential confounding
   X = np.array([0, 1, 1, 0, 1, 0, 1])  # Treatment
   Y = np.array([0, 1, 0, 0, 1, 1, 1])  # Outcome
   
   scenario = BinaryConf(X, Y)

**Available algorithms**:
   - **ATE**: manski, tianpearl, entropybounds, causaloptim, zaffalonbounds, autobound
   - **PNS**: tianpearl, entropybounds, causaloptim, zaffalonbounds, autobound

BinaryIV: Binary Instrumental Variable Setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**: When you have a valid instrumental variable that affects treatment but not outcome directly.

**Assumptions**:
   - Binary instrument, treatment, and outcome
   - Instrument affects treatment (relevance)
   - Instrument doesn't affect outcome directly (exclusion restriction)
   - Instrument is unconfounded (exogeneity)

**Causal graph**:

.. code-block:: text

   Z → X → Y
       ↑   ↑
        U (unmeasured)

**Example**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryIV
   import numpy as np
   
   # IV data
   Z = np.array([0, 1, 1, 0, 1, 0, 1])  # Instrument  
   X = np.array([0, 1, 1, 0, 1, 0, 0])  # Treatment (influenced by Z)
   Y = np.array([0, 1, 0, 0, 1, 1, 0])  # Outcome
   
   scenario = BinaryIV(X, Y, Z)

**Available algorithms**:
   - **ATE**: causaloptim, zaffalonbounds, autobound
   - **PNS**: causaloptim, zaffalonbounds, autobound

ContIV: Continuous Instrumental Variable Setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**: When you have continuous variables and a valid instrument.

**Assumptions**:
   - Continuous instrument, treatment, and outcome
   - Standard IV assumptions hold

**Example**:

.. code-block:: python

   from causalboundingengine.scenarios import ContIV
   import numpy as np
   
   # Continuous IV data (will be discretized)
   Z = np.random.normal(0, 1, 100)  # Continuous instrument
   X = Z + np.random.normal(0, 0.5, 100)  # Treatment
   Y = X + np.random.normal(0, 0.5, 100)  # Outcome
   
   scenario = ContIV(X, Y, Z)

**Available algorithms**:
   - **ATE**: zhangbareinboim

Algorithm Categories
--------------------

Algorithms in CausalBoundingEngine can be categorized by their approach and requirements:

Pure Python Algorithms
~~~~~~~~~~~~~~~~~~~~~~~

These algorithms require only core Python dependencies:

**Manski Bounds**
   - Most conservative bounds
   - No additional assumptions
   - Fast computation
   - Only available for ATE in confounded settings

**Tian-Pearl Bounds**  
   - Nonparametric bounds
   - Uses distribution inequalities
   - Available for both ATE and PNS

**AutoBound**
   - Optimization-based approach
   - Uses linear programming
   - Handles complex causal graphs
   - Works with both confounded and IV settings

R-based Algorithms
~~~~~~~~~~~~~~~~~~

These require R installation and the ``rpy2`` package:

**CausalOptim**
   - Symbolic derivation of bounds
   - Optimization using R's ``causaloptim`` package
   - Supports both confounded and IV settings
   - Available for both ATE and PNS

Java-based Algorithms  
~~~~~~~~~~~~~~~~~~~~~

These require Java and the ``jpype1`` package:

**Zaffalonbounds**
   - Uses CREMA and CREDICI libraries
   - Credal network approach
   - EM-based learning
   - Supports both confounded and IV settings

Specialized Algorithms
~~~~~~~~~~~~~~~~~~~~~~

**EntropyBounds**
   - Based on mutual information constraints
   - Requires theta parameter (information constraint level)
   - Convex optimization approach

**ZhangBareinboim**
   - Designed for continuous IV settings
   - Linear programming formulation
   - Handles compliance types

Using the API
--------------

The CausalBoundingEngine API is designed for simplicity and consistency across all algorithms and scenarios.

Basic Pattern
~~~~~~~~~~~~~

.. code-block:: python

   # 1. Import scenario
   from causalboundingengine.scenarios import BinaryConf
   
   # 2. Create scenario with data
   scenario = BinaryConf(X, Y, Z)  # Z optional
   
   # 3. Compute bounds
   ate_bounds = scenario.ATE.algorithm_name()
   pns_bounds = scenario.PNS.algorithm_name()

Dynamic Algorithm Access
~~~~~~~~~~~~~~~~~~~~~~~~

Algorithms are accessed dynamically through the scenario's ATE and PNS dispatchers:

.. code-block:: python

   scenario = BinaryConf(X, Y)
   
   # These are equivalent:
   bounds1 = scenario.ATE.manski()
   
   algorithm_name = 'manski'
   bounds2 = getattr(scenario.ATE, algorithm_name)()

Algorithm Discovery
~~~~~~~~~~~~~~~~~~~

Find available algorithms programmatically:

.. code-block:: python

   scenario = BinaryConf(X, Y)
   
   # Get available algorithms for each query type
   ate_algorithms = scenario.get_algorithms('ATE')
   pns_algorithms = scenario.get_algorithms('PNS')
   
   print(f"ATE algorithms: {ate_algorithms}")
   print(f"PNS algorithms: {pns_algorithms}")

Algorithm Parameters
~~~~~~~~~~~~~~~~~~~~

Some algorithms accept additional parameters:

.. code-block:: python

   # EntropyBounds with information constraint
   bounds = scenario.ATE.entropybounds(theta=0.5)
   
   # CausalOptim with custom R path
   bounds = scenario.ATE.causaloptim(r_path="/custom/path/to/R")

Error Handling
~~~~~~~~~~~~~~

The framework provides graceful error handling:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.WARNING)
   
   # Missing dependencies are handled gracefully
   try:
       bounds = scenario.ATE.causaloptim()
   except ImportError as e:
       print(f"Algorithm not available: {e}")
   
   # Failed computations return trivial bounds
   bounds = scenario.ATE.some_algorithm()  # May return (-1, 1) or (0, 1) on failure

Data Handling
--------------

Data Validation
~~~~~~~~~~~~~~~

CausalBoundingEngine expects binary data (0s and 1s):

.. code-block:: python

   import numpy as np
   
   # Good: Binary data
   X = np.array([0, 1, 1, 0])
   Y = np.array([1, 0, 1, 1])
   
   # Avoid: Non-binary data (may cause issues)
   X = np.array([0.5, 1.2, 0.8, 0.1])  # Not recommended

Data Conversion
~~~~~~~~~~~~~~~

Convert continuous/categorical data to binary:

.. code-block:: python

   import pandas as pd
   import numpy as np
   
   # From pandas DataFrame
   df = pd.read_csv('data.csv')
   
   # Convert continuous to binary (median split)
   X_binary = (df['treatment'] > df['treatment'].median()).astype(int)
   Y_binary = (df['outcome'] > df['outcome'].median()).astype(int)
   
   # Convert categorical to binary
   Z_binary = (df['group'] == 'treatment').astype(int)

Missing Data
~~~~~~~~~~~~

Handle missing data before using CausalBoundingEngine:

.. code-block:: python

   import pandas as pd
   import numpy as np
   
   df = pd.read_csv('data.csv')
   
   # Remove rows with missing values
   df_clean = df.dropna(subset=['treatment', 'outcome'])
   
   # Or impute missing values
   df['treatment'].fillna(df['treatment'].mode()[0], inplace=True)

Best Practices
--------------

Algorithm Selection
~~~~~~~~~~~~~~~~~~~

**For robustness**: Compare multiple algorithms

.. code-block:: python

   algorithms = ['manski', 'tianpearl', 'autobound']
   results = {}
   
   for alg in algorithms:
       try:
           results[alg] = getattr(scenario.ATE, alg)()
       except:
           continue

**For conservative bounds**: Use Manski bounds

.. code-block:: python

   # Most conservative bounds (widest interval)
   conservative_bounds = scenario.ATE.manski()

**For tighter bounds**: Use algorithms with additional assumptions

.. code-block:: python

   # Tighter bounds with information constraint
   tighter_bounds = scenario.ATE.entropybounds(theta=0.1)

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fast algorithms**: Manski, Tian-Pearl
**Moderate algorithms**: EntropyBounds, AutoBound  
**Slower algorithms**: CausalOptim, Zaffalonbounds (external engines)

.. code-block:: python

   import time
   
   # Quick bounds for exploration
   start = time.time()
   quick_bounds = scenario.ATE.manski()
   print(f"Manski: {time.time() - start:.3f}s")
   
   # More sophisticated bounds for final analysis
   start = time.time()
   detailed_bounds = scenario.ATE.autobound()
   print(f"AutoBound: {time.time() - start:.3f}s")

Reproducibility
~~~~~~~~~~~~~~~

Set random seeds for reproducible results:

.. code-block:: python

   import numpy as np
   
   # Set seed for data generation
   np.random.seed(42)
   
   # Some algorithms may have internal randomness
   # Check algorithm documentation for specific seeds

Common Patterns
---------------

Pattern 1: Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test sensitivity to assumptions by varying parameters:

.. code-block:: python

   def sensitivity_analysis(X, Y, theta_values):
       scenario = BinaryConf(X, Y)
       results = []
       
       for theta in theta_values:
           bounds = scenario.ATE.entropybounds(theta=theta)
           results.append({
               'theta': theta,
               'lower': bounds[0],
               'upper': bounds[1],
               'width': bounds[1] - bounds[0]
           })
       
       return results

   # Test different information constraints
   thetas = [0.1, 0.5, 1.0, 2.0]
   sensitivity = sensitivity_analysis(X, Y, thetas)

Pattern 2: Algorithm Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare algorithm performance systematically:

.. code-block:: python

   def compare_algorithms(X, Y, algorithms=None):
       if algorithms is None:
           scenario = BinaryConf(X, Y)
           algorithms = scenario.get_algorithms('ATE')
       
       results = []
       scenario = BinaryConf(X, Y)
       
       for alg_name in algorithms:
           try:
               start_time = time.time()
               bounds = getattr(scenario.ATE, alg_name)()
               end_time = time.time()
               
               results.append({
                   'algorithm': alg_name,
                   'lower_bound': bounds[0],
                   'upper_bound': bounds[1], 
                   'width': bounds[1] - bounds[0],
                   'computation_time': end_time - start_time
               })
           except Exception as e:
               print(f"Failed {alg_name}: {e}")
       
       return pd.DataFrame(results)

Pattern 3: Bootstrap Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add uncertainty quantification:

.. code-block:: python

   def bootstrap_bounds(X, Y, algorithm='manski', n_bootstrap=100):
       scenario = BinaryConf(X, Y)
       alg_func = getattr(scenario.ATE, algorithm)
       
       n = len(X)
       bootstrap_bounds = []
       
       for _ in range(n_bootstrap):
           # Bootstrap sample
           indices = np.random.choice(n, n, replace=True)
           X_boot = X[indices]
           Y_boot = Y[indices]
           
           # Compute bounds on bootstrap sample
           scenario_boot = BinaryConf(X_boot, Y_boot)
           bounds = getattr(scenario_boot.ATE, algorithm)()
           bootstrap_bounds.append(bounds)
       
       # Compute confidence intervals
       lower_bounds = [b[0] for b in bootstrap_bounds]
       upper_bounds = [b[1] for b in bootstrap_bounds]
       
       return {
           'lower_95ci': (np.percentile(lower_bounds, 2.5), 
                         np.percentile(lower_bounds, 97.5)),
           'upper_95ci': (np.percentile(upper_bounds, 2.5),
                         np.percentile(upper_bounds, 97.5))
       }

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**"Algorithm not available"**
   - Check if optional dependencies are installed
   - Verify R/Java installation for external algorithms

**"Bounds are trivial"**
   - May indicate algorithm failure or insufficient data
   - Try different algorithms
   - Check data quality

**Memory errors with large datasets**
   - Some algorithms (especially Java-based) may need more memory
   - Consider sampling your data for exploration

**Inconsistent bounds across algorithms**
   - This is expected! Different algorithms make different assumptions
   - Document which bounds you trust more based on your domain knowledge

Getting Help
~~~~~~~~~~~~

1. Check the specific algorithm documentation
2. Verify your data meets the algorithm requirements  
3. Test with a simple synthetic dataset first
4. Check the GitHub issues page for known problems
