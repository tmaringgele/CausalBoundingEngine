Quick Start Guide
================

This guide will get you up and running with CausalBoundingEngine in minutes.

Basic Workflow
--------------

CausalBoundingEngine follows a simple three-step workflow:

1. **Prepare your data** - Treatment (X), Outcome (Y), and optional Instrument (Z)
2. **Choose a scenario** - BinaryConf, BinaryIV, or ContIV
3. **Compute bounds** - Use any available algorithm for your scenario

Example 1: Simple Confounded Setting
-------------------------------------

Let's start with a basic example where we have treatment and outcome data, but suspect unmeasured confounding:

.. code-block:: python

   import numpy as np
   from causalboundingengine.scenarios import BinaryConf
   
   # Generate some example data
   np.random.seed(42)
   n = 1000
   X = np.random.binomial(1, 0.3, n)  # Treatment (binary)
   Y = np.random.binomial(1, 0.6, n)  # Outcome (binary)
   
   # Create scenario
   scenario = BinaryConf(X, Y)
   
   # Compute ATE bounds using different algorithms
   manski_bounds = scenario.ATE.manski()
   tianpearl_bounds = scenario.ATE.tianpearl()
   
   print(f"Manski bounds: {manski_bounds}")
   print(f"Tian-Pearl bounds: {tianpearl_bounds}")

Output:

.. code-block:: text

   Manski bounds: (-0.7, 0.7)
   Tian-Pearl bounds: (-0.1, 0.9)

Available Algorithms for BinaryConf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # See all available algorithms
   print("ATE algorithms:", scenario.get_algorithms('ATE'))
   print("PNS algorithms:", scenario.get_algorithms('PNS'))

.. code-block:: text

   ATE algorithms: ['manski', 'tianpearl', 'entropybounds', 'causaloptim', 'zaffalonbounds', 'autobound']
   PNS algorithms: ['tianpearl', 'entropybounds', 'causaloptim', 'zaffalonbounds', 'autobound']

Example 2: Instrumental Variable Setting
-----------------------------------------

When you have an instrumental variable, use the BinaryIV scenario:

.. code-block:: python

   import numpy as np
   from causalboundingengine.scenarios import BinaryIV
   
   # Generate IV data
   np.random.seed(123)
   n = 1000
   Z = np.random.binomial(1, 0.5, n)    # Instrument
   X = np.random.binomial(1, 0.3 + 0.2*Z, n)  # Treatment influenced by Z
   Y = np.random.binomial(1, 0.4 + 0.1*X, n)  # Outcome influenced by X
   
   # Create IV scenario
   scenario = BinaryIV(X, Y, Z)
   
   # Compute bounds (fewer algorithms available for IV)
   autobound_bounds = scenario.ATE.autobound()
   
   print(f"Autobound ATE: {autobound_bounds}")

Example 3: Algorithm Parameters
-------------------------------

Some algorithms accept additional parameters:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   # Data
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = BinaryConf(X, Y)
   
   # EntropyBounds with different theta values
   bounds_strict = scenario.ATE.entropybounds(theta=0.1)  # Strict constraint
   bounds_loose = scenario.ATE.entropybounds(theta=1.0)   # Loose constraint
   
   print(f"Strict bounds (θ=0.1): {bounds_strict}")
   print(f"Loose bounds (θ=1.0): {bounds_loose}")

Example 4: R-based Algorithms
------------------------------

If you have R installed with the ``r`` extra:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = BinaryConf(X, Y)
   
   try:
       # R-based Causaloptim algorithm
       bounds = scenario.ATE.causaloptim()
       print(f"Causaloptim bounds: {bounds}")
   except ImportError as e:
       print(f"R support not available: {e}")
       print("Install with: pip install causalboundingengine[r]")

Example 5: Java-based Algorithms
---------------------------------

If you have Java installed with the ``java`` extra:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = BinaryConf(X, Y)
   
   try:
       # Java-based Zaffalonbounds algorithm
       bounds = scenario.ATE.zaffalonbounds()
       print(f"Zaffalonbounds: {bounds}")
   except ImportError as e:
       print(f"Java support not available: {e}")
       print("Install with: pip install causalboundingengine[java]")

Example 6: Probability of Necessity and Sufficiency (PNS)
----------------------------------------------------------

PNS measures the probability that treatment is both necessary and sufficient for the outcome:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   # Data where treatment seems more impactful
   X = np.array([0, 0, 0, 1, 1, 1])
   Y = np.array([0, 0, 1, 1, 1, 1])  # Higher Y when X=1
   
   scenario = BinaryConf(X, Y)
   
   # Compute PNS bounds
   tianpearl_pns = scenario.PNS.tianpearl()
   entropy_pns = scenario.PNS.entropybounds(theta=0.5)
   
   print(f"Tian-Pearl PNS: {tianpearl_pns}")
   print(f"Entropy PNS (θ=0.5): {entropy_pns}")

Working with Real Data
----------------------

Loading from pandas DataFrame:

.. code-block:: python

   import pandas as pd
   from causalboundingengine.scenarios import BinaryConf
   
   # Load your data
   df = pd.read_csv('your_data.csv')
   
   # Extract variables
   X = df['treatment'].values
   Y = df['outcome'].values
   
   # Optional: convert to binary if needed
   X = (X > X.median()).astype(int)
   Y = (Y > Y.median()).astype(int)
   
   # Analyze
   scenario = BinaryConf(X, Y)
   bounds = scenario.ATE.manski()

Comparing Multiple Algorithms
------------------------------

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   import pandas as pd
   
   # Generate data
   np.random.seed(42)
   X = np.random.binomial(1, 0.4, 500)
   Y = np.random.binomial(1, 0.3 + 0.2*X, 500)
   
   scenario = BinaryConf(X, Y)
   
   # Compare multiple algorithms
   algorithms = ['manski', 'tianpearl', 'autobound']
   results = []
   
   for alg_name in algorithms:
       try:
           alg = getattr(scenario.ATE, alg_name)
           bounds = alg()
           results.append({
               'algorithm': alg_name,
               'lower': bounds[0], 
               'upper': bounds[1],
               'width': bounds[1] - bounds[0]
           })
       except Exception as e:
           print(f"Failed to run {alg_name}: {e}")
   
   # Display results
   df_results = pd.DataFrame(results)
   print(df_results)

Next Steps
----------

- Read the :doc:`core_concepts`
- Check :doc:`algorithms` for algorithm-specific documentation  
- See :doc:`examples` for more complex use cases
- Learn how to add your own algorithms in :doc:`extending`

Common Patterns
---------------

**Pattern 1: Algorithm Availability Check**

.. code-block:: python

   def safe_compute_bounds(scenario, algorithm_name, query='ATE', **kwargs):
       """Safely compute bounds with fallback."""
       try:
           dispatcher = getattr(scenario, query)
           algorithm = getattr(dispatcher, algorithm_name)
           return algorithm(**kwargs)
       except (AttributeError, ImportError) as e:
           print(f"Algorithm {algorithm_name} not available: {e}")
           return None

**Pattern 2: Batch Processing**

.. code-block:: python

   def process_multiple_datasets(datasets, algorithm='manski'):
       """Process multiple datasets with same algorithm."""
       results = []
       for i, (X, Y) in enumerate(datasets):
           scenario = BinaryConf(X, Y)
           bounds = getattr(scenario.ATE, algorithm)()
           results.append({
               'dataset': i,
               'lower_bound': bounds[0],
               'upper_bound': bounds[1]
           })
       return results

**Pattern 3: Robustness Checking**

.. code-block:: python

   def robustness_check(X, Y, algorithms=None):
       """Check robustness across multiple algorithms."""
       if algorithms is None:
           algorithms = ['manski', 'tianpearl', 'autobound']
       
       scenario = BinaryConf(X, Y)
       bounds_list = []
       
       for alg in algorithms:
           try:
               bounds = getattr(scenario.ATE, alg)()
               bounds_list.append(bounds)
           except:
               continue
       
       if bounds_list:
           all_lowers = [b[0] for b in bounds_list]
           all_uppers = [b[1] for b in bounds_list]
           return (min(all_lowers), max(all_uppers))
       return None
