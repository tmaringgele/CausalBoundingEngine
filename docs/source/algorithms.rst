Algorithms Reference
===================

This page provides detailed documentation for all algorithms available in CausalBoundingEngine.

.. note::
   All algorithms are implementations of published research. For complete citations, references, and proper attribution, see the :doc:`references` page. Please cite the original papers when using these algorithms in your research.

Algorithm Overview
------------------

.. list-table:: Algorithm Comparison
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Algorithm
     - ATE
     - PNS
     - Scenarios
     - Dependencies
     - Notes
   * - Manski
     - ✓
     - ✗
     - BinaryConf
     - Core
     - Most conservative
   * - TianPearl
     - ✓
     - ✓
     - BinaryConf
     - Core
     - Nonparametric
   * - Autobound
     - ✓
     - ✓
     - BinaryConf, BinaryIV
     - Core
     - Optimization-based
   * - EntropyBounds
     - ✓
     - ✓
     - BinaryConf
     - Core
     - Requires theta parameter
   * - Causaloptim
     - ✓
     - ✓
     - BinaryConf, BinaryIV
     - R
     - Symbolic derivation
   * - Zaffalonbounds
     - ✓
     - ✓
     - BinaryConf, BinaryIV
     - Java
     - Credal networks
   * - ZhangBareinboim
     - ✓
     - ✗
     - ContIV
     - Core
     - Continuous IV

Core Algorithms
---------------

Manski Bounds
~~~~~~~~~~~~~

**Reference**: Manski, C. F. (1990). Nonparametric Bounds on Treatment Effects. *The American Economic Review*, 80(2), 319-323. See :doc:`references` for complete citation.

**Description**: Provides the most conservative bounds on the ATE under no additional assumptions beyond the observed data. These bounds are derived by considering the worst-case scenarios for the unobserved potential outcomes.

**Mathematical Foundation**:

For binary outcomes, the ATE bounds are:

.. math::
   
   ATE_{lower} = \max(P(Y=1|X=1) - P(Y=1|X=0) - 1, -1)
   
   ATE_{upper} = \min(P(Y=1|X=1) - P(Y=1|X=0) + 1, 1)

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = BinaryConf(X, Y)
   
   bounds = scenario.ATE.manski()
   print(f"Manski bounds: {bounds}")

**Properties**:
   - No additional assumptions required
   - Always provides valid bounds
   - Most conservative (widest intervals)
   - Fast computation
   - Only available for ATE

**When to use**:
   - As a baseline for comparison
   - When no additional assumptions can be justified
   - For quick exploratory analysis

Tian-Pearl Bounds
~~~~~~~~~~~~~~~~~

**Reference**: Tian, J., & Pearl, J. (2000). Probabilities of Causation: Bounds and Identification. *Annals of Mathematics and Artificial Intelligence*, 28(1-4), 287-313. See :doc:`references` for complete citation.

**Description**: Nonparametric bounds that use the joint distribution of treatment and outcome to derive tighter bounds than Manski, particularly for PNS.

**Mathematical Foundation**:

For ATE:

.. math::
   
   ATE_{lower} = P(Y=1|X=1) - (1 - P(Y=1|X=0))
   
   ATE_{upper} = (1 - P(Y=1|X=0)) - P(Y=1|X=1)

For PNS:

.. math::
   
   PNS_{lower} = 0
   
   PNS_{upper} = P(Y=1, X=1) + P(Y=0, X=0)

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = BinaryConf(X, Y)
   
   ate_bounds = scenario.ATE.tianpearl()
   pns_bounds = scenario.PNS.tianpearl()
   print(f"Tian-Pearl ATE: {ate_bounds}")
   print(f"Tian-Pearl PNS: {pns_bounds}")

**Properties**:
   - Often tighter than Manski bounds
   - Available for both ATE and PNS
   - Fast computation
   - No additional parameters

**When to use**:
   - When you want tighter bounds than Manski
   - For PNS estimation
   - As a standard nonparametric approach

Autobound
~~~~~~~~~

**Reference**: Duarte, G., Finkelstein, N., Knox, D., Mummolo, J., & Shpitser, I. (2023). An Automated Approach to Causal Inference in Discrete Settings. *Journal of the American Statistical Association*, 1-12. See :doc:`references` for complete citation.

**Description**: A general-purpose algorithm that formulates causal bounding as a linear programming problem. Can handle complex causal graphs and both confounded and IV settings.

**Mathematical Foundation**:

Autobound represents the causal problem using:
   - Decision variables for each potential outcome type
   - Constraints matching observed distributions
   - Linear programming optimization

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf, BinaryIV
   import numpy as np
   
   # Confounded setting
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = BinaryConf(X, Y)
   bounds = scenario.ATE.autobound()
   
   # IV setting
   Z = np.array([0, 1, 1, 0, 1])
   scenario_iv = BinaryIV(X, Y, Z)
   bounds_iv = scenario_iv.ATE.autobound()

**Properties**:
   - Works with both confounded and IV settings
   - Available for both ATE and PNS
   - Principled optimization approach
   - Moderate computation time

**When to use**:
   - When you need a general-purpose algorithm
   - For IV settings where other algorithms aren't available
   - When you want theoretically grounded bounds

EntropyBounds
~~~~~~~~~~~~~

**Reference**: Jiang, Z., & Shpitser, I. (2020). Approximate Causal Effect Identification under Weak Confounding. *Proceedings of the 37th International Conference on Machine Learning*, 4740-4750. See :doc:`references` for complete citation.

**Description**: Uses mutual information constraints to bound causal effects under the assumption of "weak confounding" - limited dependence between confounders and observed variables.

**Mathematical Foundation**:

The algorithm constrains the mutual information between potential outcomes and treatment:

.. math::
   
   I(Y(0), Y(1); X) \leq \theta

where θ is a user-specified parameter controlling the strength of confounding.

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = BinaryConf(X, Y)
   
   # Different theta values give different bounds
   strict_bounds = scenario.ATE.entropybounds(theta=0.1)  # Strong assumption
   loose_bounds = scenario.ATE.entropybounds(theta=1.0)   # Weak assumption
   
   print(f"Strict bounds (θ=0.1): {strict_bounds}")
   print(f"Loose bounds (θ=1.0): {loose_bounds}")

**Parameters**:
   - **theta** (float): Information constraint level. Lower values give tighter bounds but require stronger assumptions.

**Properties**:
   - Requires theta parameter (no default)
   - Available for both ATE and PNS
   - Uses convex optimization
   - Sensitive to theta choice

**When to use**:
   - When you can justify weak confounding assumptions
   - For sensitivity analysis across different theta values
   - When domain knowledge suggests limited confounding

External Engine Algorithms
---------------------------

Causaloptim
~~~~~~~~~~~

**Dependencies**: R, rpy2, causaloptim R package

**Reference**: Sachs, M. C., Sjölander, A., & Gabriel, E. E. (2022). A General Method for Deriving Tight Symbolic Bounds on Causal Effects. *Journal of Computational and Graphical Statistics*, 31(2), 496-510. See :doc:`references` for complete citation.

**Description**: Uses symbolic computation to derive analytic bounds on causal effects. Integrates with the R package ``causaloptim`` for graph specification and optimization.

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf, BinaryIV
   import numpy as np
   
   # Confounded setting
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = BinaryConf(X, Y)
   
   try:
       bounds = scenario.ATE.causaloptim()
       print(f"Causaloptim bounds: {bounds}")
   except ImportError:
       print("R support not available")
   
   # IV setting
   Z = np.array([0, 1, 1, 0, 1])
   scenario_iv = BinaryIV(X, Y, Z)
   bounds_iv = scenario_iv.ATE.causaloptim()

**Parameters**:
   - **r_path** (str, optional): Custom path to R executable

**Properties**:
   - Symbolic derivation of bounds
   - Works with both confounded and IV settings
   - Available for both ATE and PNS
   - Requires R installation

**Installation**:

.. code-block:: bash

   # Install R support
   pip install causalboundingengine[r]

**When to use**:
   - When you want symbolically derived bounds
   - For complex causal graphs
   - When R environment is available

Zaffalonbounds
~~~~~~~~~~~~~~

**Dependencies**: Java, jpype1, CREMA/CREDICI libraries

**Reference**: Zaffalon, M., Antonucci, A., Cabañas, R., Huber, D., & Azzimonti, D. (2022). Bounding Counterfactuals under Selection Bias. *Proceedings of The 11th International Conference on Probabilistic Graphical Models*, 289-300. Uses CREMA and CREDICI libraries. See :doc:`references` for complete citation.

**Description**: Uses credal networks and EM-based learning to compute bounds. Based on the CREMA and CREDICI Java libraries developed at IDSIA.

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import BinaryConf, BinaryIV
   import numpy as np
   
   # Confounded setting
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = BinaryConf(X, Y)
   
   try:
       bounds = scenario.ATE.zaffalonbounds()
       print(f"Zaffalonbounds: {bounds}")
   except ImportError:
       print("Java support not available")

**Properties**:
   - Uses credal network inference
   - EM-based parameter learning
   - Works with both confounded and IV settings
   - Available for both ATE and PNS
   - Requires Java installation

**Installation**:

.. code-block:: bash

   # Install Java support
   pip install causalboundingengine[java]

**When to use**:
   - When you want Bayesian-style bounds
   - For complex probabilistic reasoning
   - When Java environment is available

Specialized Algorithms
----------------------

ZhangBareinboim
~~~~~~~~~~~~~~~

**Reference**: Zhang, J., & Bareinboim, E. (2021). Bounding Causal Effects on Continuous Outcome. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(13), 12207-12215. See :doc:`references` for complete citation.

**Description**: Designed specifically for continuous instrumental variable settings. Uses linear programming to handle compliance types in IV analysis.

**Usage**:

.. code-block:: python

   from causalboundingengine.scenarios import ContIV
   import numpy as np
   
   # Continuous data (will be discretized internally)
   Z = np.random.normal(0, 1, 100)  # Instrument
   X = Z + np.random.normal(0, 0.5, 100)  # Treatment
   Y = X + np.random.normal(0, 0.5, 100)  # Outcome
   
   scenario = ContIV(X, Y, Z)
   bounds = scenario.ATE.zhangbareinboim()

**Properties**:
   - Specifically for continuous IV settings
   - Handles compliance types automatically
   - Only available for ATE
   - Uses linear programming

**When to use**:
   - With continuous instrumental variables
   - When compliance patterns are complex
   - For rigorous IV analysis

Algorithm Implementation Details
--------------------------------

Error Handling
~~~~~~~~~~~~~~

All algorithms implement consistent error handling:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.WARNING)
   
   # Failed algorithms return trivial bounds
   scenario = BinaryConf(X, Y)
   bounds = scenario.ATE.some_algorithm()
   
   # Check for trivial bounds
   if bounds == (-1.0, 1.0):  # ATE trivial bounds
       print("Algorithm failed, returned trivial bounds")
   
   if bounds == (0.0, 1.0):   # PNS trivial bounds
       print("Algorithm failed, returned trivial bounds")

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Typical Performance
   :header-rows: 1
   :widths: 30 20 50

   * - Algorithm
     - Speed
     - Notes
   * - Manski
     - Very Fast
     - Simple calculations
   * - TianPearl
     - Very Fast
     - Simple calculations
   * - Autobound
     - Moderate
     - Linear programming
   * - EntropyBounds
     - Moderate
     - Convex optimization
   * - Causaloptim
     - Slow
     - R interface overhead
   * - Zaffalonbounds
     - Very Slow
     - Java interface + EM algorithm
   * - ZhangBareinboim
     - Moderate
     - Linear programming

Memory Usage
~~~~~~~~~~~~

Most algorithms have modest memory requirements, but some considerations:

- **Zaffalonbounds**: May need increased JVM heap size for large datasets
- **Autobound**: Linear programming may use significant memory
- **EntropyBounds**: Convex optimization scales with data size

.. code-block:: python

   # For large datasets with Java algorithms
   import jpype
   jpype.startJVM("-Xmx4g")  # 4GB heap size

Choosing the Right Algorithm
----------------------------

Decision Tree
~~~~~~~~~~~~~

1. **What type of data do you have?**
   
   - Binary treatment/outcome → Continue to step 2
   - Continuous variables → Use ZhangBareinboim (if IV available)

2. **Do you have an instrument?**
   
   - Yes → Use Autobound, Causaloptim, or Zaffalonbounds
   - No → Continue to step 3

3. **What are your computational constraints?**
   
   - Need fast results → Use Manski or TianPearl
   - Have more time → Consider Autobound, Causaloptim, or Zaffalonbounds

4. **What (further) assumptions can you make?**
   
   - Weak confounding → Use EntropyBounds with appropriate theta

5. **What external dependencies do you have?**
   
   - Core Python only → Use Manski, TianPearl, Autobound, or EntropyBounds
   - R available → Consider Causaloptim
   - Java available → Consider Zaffalonbounds

Robustness Strategy
~~~~~~~~~~~~~~~~~~~

For important analyses, consider using multiple algorithms:

.. code-block:: python

   def robust_analysis(X, Y, Z=None):
       \"\"\"Run multiple algorithms for robustness.\"\"\"
       if Z is None:
           scenario = BinaryConf(X, Y)
           algorithms = ['manski', 'tianpearl', 'autobound']
       else:
           scenario = BinaryIV(X, Y, Z)
           algorithms = ['autobound', 'causaloptim', 'zaffalonbounds']
       
       results = {}
       for alg in algorithms:
           try:
               results[alg] = getattr(scenario.ATE, alg)()
           except Exception as e:
               print(f"Failed {alg}: {e}")
       
       return results

   # Compare results
   bounds_dict = robust_analysis(X, Y)
   for alg, bounds in bounds_dict.items():
       print(f"{alg}: {bounds}")

This approach helps identify:
   - Consensus across methods
   - Algorithms that may be failing
   - Sensitivity to different assumptions
