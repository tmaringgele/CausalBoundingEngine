Examples
========

This page provides comprehensive examples of using CausalBoundingEngine in various scenarios.

Basic Examples
--------------

Example 1: Getting Started
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple analysis with binary confounded data:

.. code-block:: python

   import numpy as np
   from causalboundingengine.scenarios import BinaryConf
   
   # Generate synthetic data
   np.random.seed(42)
   n = 1000
   
   # Simulate confounded treatment assignment
   # U is an unmeasured confounder
   U = np.random.binomial(1, 0.5, n)  # Hidden confounder
   X = np.random.binomial(1, 0.3 + 0.4 * U, n)  # Treatment depends on U
   Y = np.random.binomial(1, 0.2 + 0.3 * X + 0.2 * U, n)  # Outcome depends on X and U
   
   # Create scenario (U is not observed)
   scenario = BinaryConf(X, Y)
   
   # Compute bounds using different algorithms
   print("=== ATE Bounds ===")
   print(f"Manski:     {scenario.ATE.manski()}")
   print(f"Autobound:  {scenario.ATE.autobound()}")
   
   print("\\n=== PNS Bounds ===")
   print(f"Tian-Pearl: {scenario.PNS.tianpearl()}")
   print(f"Autobound:  {scenario.PNS.autobound()}")

Example 2: Instrumental Variable Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using an instrumental variable to get tighter bounds:

.. code-block:: python

   import numpy as np
   from causalboundingengine.scenarios import BinaryIV
   
   # Generate IV data
   np.random.seed(123)
   n = 500
   
   # Valid instrument (random assignment)
   Z = np.random.binomial(1, 0.5, n)
   
   # Unmeasured confounder
   U = np.random.binomial(1, 0.4, n)
   
   # Treatment: influenced by instrument and confounder
   X = np.random.binomial(1, 0.2 + 0.3 * Z + 0.3 * U, n)
   
   # Outcome: influenced by treatment and confounder (not directly by instrument)
   Y = np.random.binomial(1, 0.3 + 0.4 * X + 0.2 * U, n)
   
   # Create IV scenario
   scenario = BinaryIV(X, Y, Z)
   
   print("=== IV Analysis ===")
   print(f"Autobound ATE:  {scenario.ATE.autobound()}")
   print(f"Autobound PNS:  {scenario.PNS.autobound()}")
   
   # Compare with confounded scenario (ignore instrument)
   scenario_conf = BinaryConf(X, Y)
   print(f"\\nWithout IV - Manski ATE: {scenario_conf.ATE.manski()}")
   print("(IV should give tighter bounds)")

Example 3: Algorithm Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Systematic comparison of multiple algorithms:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from causalboundingengine.scenarios import BinaryConf
   import time
   
   # Generate data
   np.random.seed(42)
   X = np.random.binomial(1, 0.4, 200)
   Y = np.random.binomial(1, 0.3 + 0.2 * X, 200)
   scenario = BinaryConf(X, Y)
   
   # Compare algorithms
   algorithms = ['manski', 'autobound']
   results = []
   
   for alg_name in algorithms:
       print(f"Running {alg_name}...")
       
       start_time = time.time()
       try:
           alg_func = getattr(scenario.ATE, alg_name)
           bounds = alg_func()
           success = True
           error_msg = None
       except Exception as e:
           bounds = (None, None)
           success = False
           error_msg = str(e)
       end_time = time.time()
       
       results.append({
           'algorithm': alg_name,
           'lower_bound': bounds[0],
           'upper_bound': bounds[1],
           'width': bounds[1] - bounds[0] if success else None,
           'time_seconds': end_time - start_time,
           'success': success,
           'error': error_msg
       })
   
   # Display results
   df = pd.DataFrame(results)
   print("\\n=== Algorithm Comparison ===")
   print(df.to_string(index=False))

Advanced Examples
-----------------

Example 4: Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Testing sensitivity to different assumptions using EntropyBounds:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from causalboundingengine.scenarios import BinaryConf
   
   # Generate data with moderate confounding
   np.random.seed(42)
   n = 500
   U = np.random.binomial(1, 0.5, n)
   X = np.random.binomial(1, 0.3 + 0.4 * U, n)
   Y = np.random.binomial(1, 0.2 + 0.3 * X + 0.3 * U, n)
   
   scenario = BinaryConf(X, Y)
   
   # Test different theta values (information constraints)
   theta_values = [0.05, 0.1, 0.2, 0.5, 0.8, 0.95]
   results = []
   
   for theta in theta_values:
       try:
           bounds = scenario.ATE.entropybounds(theta=theta)
           results.append({
               'theta': theta,
               'lower': bounds[0],
               'upper': bounds[1],
               'width': bounds[1] - bounds[0]
           })
       except Exception as e:
           print(f"Failed for theta={theta}: {e}")
   
   # Display results
   df_sensitivity = pd.DataFrame(results)
   print("=== Sensitivity Analysis ===")
   print(df_sensitivity.to_string(index=False))
   
   # Plot bounds vs theta
   plt.figure(figsize=(10, 6))
   plt.plot(df_sensitivity['theta'], df_sensitivity['lower'], 'bo-', label='Lower bound')
   plt.plot(df_sensitivity['theta'], df_sensitivity['upper'], 'ro-', label='Upper bound')
   plt.fill_between(df_sensitivity['theta'], df_sensitivity['lower'], 
                    df_sensitivity['upper'], alpha=0.3, color='gray')
   plt.xlabel('Theta (Information Constraint)')
   plt.ylabel('ATE Bounds')
   plt.title('Sensitivity to Information Constraint')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Example 5: Bootstrap Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding uncertainty quantification using bootstrap:

.. code-block:: python

   import numpy as np
   from causalboundingengine.scenarios import BinaryConf
   import pandas as pd
   
   def bootstrap_bounds(X, Y, algorithm='manski', n_bootstrap=200, alpha=0.05):
       n = len(X)
       bootstrap_results = []
       
       for i in range(n_bootstrap):
           # Bootstrap sample
           indices = np.random.choice(n, n, replace=True)
           X_boot = X[indices]
           Y_boot = Y[indices]
           
           # Compute bounds
           scenario_boot = BinaryConf(X_boot, Y_boot)
           alg_func = getattr(scenario_boot.ATE, algorithm)
           bounds = alg_func()
           
           bootstrap_results.append({
               'iteration': i,
               'lower': bounds[0],
               'upper': bounds[1]
           })
       
       # Compute confidence intervals
       df_boot = pd.DataFrame(bootstrap_results)
       
       lower_ci = (
           np.percentile(df_boot['lower'], 100 * alpha/2),
           np.percentile(df_boot['lower'], 100 * (1 - alpha/2))
       )
       upper_ci = (
           np.percentile(df_boot['upper'], 100 * alpha/2),
           np.percentile(df_boot['upper'], 100 * (1 - alpha/2))
       )
       
       return {
           'bootstrap_samples': df_boot,
           'lower_bound_ci': lower_ci,
           'upper_bound_ci': upper_ci,
           'alpha': alpha
       }
   
   # Generate data
   np.random.seed(42)
   X = np.random.binomial(1, 0.4, 100)
   Y = np.random.binomial(1, 0.3 + 0.3 * X, 100)
   
   # Bootstrap analysis
   print("Computing bootstrap confidence intervals...")
   boot_results = bootstrap_bounds(X, Y, algorithm='manski', n_bootstrap=100)
   
   # Original bounds
   scenario = BinaryConf(X, Y)
   original_bounds = scenario.ATE.manski()
   
   print("=== Bootstrap Results ===")
   print(f"Original bounds: {original_bounds}")
   print(f"Lower bound 95% CI: {boot_results['lower_bound_ci']}")
   print(f"Upper bound 95% CI: {boot_results['upper_bound_ci']}")

Real-World Examples
-------------------

Example 6: Medical Treatment Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzing treatment effectiveness with potential confounding:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from causalboundingengine.scenarios import BinaryConf
   
   # Simulate medical data
   np.random.seed(42)
   n = 800
   
   # Patient characteristics (unmeasured severity)
   severity = np.random.beta(2, 5, n)  # Most patients have low severity
   
   # Treatment assignment (more severe patients more likely to receive treatment)
   treatment_prob = 0.3 + 0.4 * (severity > 0.5)
   X = np.random.binomial(1, treatment_prob, n)
   
   # Recovery outcome (depends on treatment and severity)
   recovery_prob = 0.4 + 0.3 * X - 0.2 * severity
   recovery_prob = np.clip(recovery_prob, 0.05, 0.95)  # Keep probabilities valid
   Y = np.random.binomial(1, recovery_prob, n)
   
   # Create DataFrame for analysis
   df = pd.DataFrame({
       'treatment': X,
       'recovery': Y,
       'severity': severity  # This would be unmeasured in practice
   })
   
   print("=== Medical Treatment Analysis ===")
   print(f"Sample size: {n}")
   print(f"Treatment rate: {np.mean(X):.3f}")
   print(f"Recovery rate: {np.mean(Y):.3f}")
   print(f"Recovery rate | Treated: {np.mean(Y[X==1]):.3f}")
   print(f"Recovery rate | Control: {np.mean(Y[X==0]):.3f}")
   print(f"Naive ATE estimate: {np.mean(Y[X==1]) - np.mean(Y[X==0]):.3f}")
   
   # Causal bounds analysis (ignoring severity as it's unmeasured)
   scenario = BinaryConf(X, Y)
   
   print("\\n=== Causal Bounds (accounting for unmeasured confounding) ===")
   print(f"Manski bounds:     {scenario.ATE.manski()}")
   print(f"Autobound:         {scenario.ATE.autobound()}")
   
   # True ATE (if we could observe severity)
   true_ate = np.mean(recovery_prob * 1 - (recovery_prob - 0.3))  # Approximate
   print(f"\\nApproximate true ATE: {true_ate:.3f}")

Example 7: Economic Policy Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluating a job training program with instrumental variable:

.. code-block:: python

   import numpy as np
   from causalboundingengine.scenarios import BinaryIV, BinaryConf
   
   # Simulate job training program evaluation
   np.random.seed(123)
   n = 1000
   
   # Random assignment to training eligibility (instrument)
   eligible = np.random.binomial(1, 0.5, n)
   
   # Individual motivation (unmeasured confounder)
   motivation = np.random.beta(2, 3, n)
   
   # Training participation (influenced by eligibility and motivation)
   participation_prob = 0.2 + 0.5 * eligible + 0.3 * motivation
   participation_prob = np.clip(participation_prob, 0.05, 0.95)
   training = np.random.binomial(1, participation_prob, n)
   
   # Employment outcome (influenced by training and motivation)
   employment_prob = 0.4 + 0.25 * training + 0.2 * motivation
   employment_prob = np.clip(employment_prob, 0.05, 0.95)
   employed = np.random.binomial(1, employment_prob, n)
   
   print("=== Job Training Program Evaluation ===")
   print(f"Eligibility rate: {np.mean(eligible):.3f}")
   print(f"Training participation rate: {np.mean(training):.3f}")
   print(f"Employment rate: {np.mean(employed):.3f}")
   print(f"Compliance rate: {np.mean(training[eligible==1]):.3f}")
   
   # IV Analysis
   scenario_iv = BinaryIV(training, employed, eligible)
   print("\\n=== IV Bounds ===")
   print(f"Autobound ATE: {scenario_iv.ATE.autobound()}")
   
   # Compare with confounded analysis
   scenario_conf = BinaryConf(training, employed)
   print("\\n=== Confounded Analysis (no IV) ===")
   print(f"Manski bounds: {scenario_conf.ATE.manski()}")
   print("(IV bounds should be tighter if instrument is valid)")

Example 8: Large-Scale Comparison Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive analysis across multiple datasets:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from causalboundingengine.scenarios import BinaryConf
    import time

    def generate_dataset(n, confounding_strength=0.5, seed=None):
        if seed is not None:
            np.random.seed(seed)

        U = np.random.binomial(1, 0.5, n)
        X_prob = np.clip(0.3 + confounding_strength * U, 0, 1)
        X = np.random.binomial(1, X_prob, n)
        Y_prob = np.clip(0.2 + 0.3 * X + confounding_strength * U, 0, 1)
        Y = np.random.binomial(1, Y_prob, n)

        return X, Y

    def analyze_dataset(X, Y, dataset_id):
        scenario = BinaryConf(X, Y)
        algorithms = ['manski', 'autobound']

        results = []
        for alg_name in algorithms:
            start_time = time.time()
            try:
                alg_func = getattr(scenario.ATE, alg_name)
                bounds = alg_func()
                end_time = time.time()

                results.append({
                    'dataset_id': dataset_id,
                    'algorithm': alg_name,
                    'lower_bound': bounds[0],
                    'upper_bound': bounds[1],
                    'width': bounds[1] - bounds[0],
                    'computation_time': end_time - start_time,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'dataset_id': dataset_id,
                    'algorithm': alg_name,
                    'lower_bound': None,
                    'upper_bound': None,
                    'width': None,
                    'computation_time': None,
                    'success': False
                })

        return results

    # Run comparison study
    print("=== Large-Scale Comparison Study ===")

    # Generate multiple datasets
    datasets = []
    dataset_configs = [
        {'n': 100, 'confounding': 0.2, 'name': 'Small, Weak confounding'},
        {'n': 100, 'confounding': 0.8, 'name': 'Small, Strong confounding'},
        {'n': 1000, 'confounding': 0.2, 'name': 'Large, Weak confounding'},
        {'n': 1000, 'confounding': 0.8, 'name': 'Large, Strong confounding'},
    ]

    all_results = []
    for i, config in enumerate(dataset_configs):
        print(f"Analyzing dataset {i+1}: {config['name']}")

        X, Y = generate_dataset(
            n=config['n'],
            confounding_strength=config['confounding'],
            seed=42 + i
        )

        dataset_results = analyze_dataset(X, Y, i+1)

        # Add dataset metadata
        for result in dataset_results:
            result.update({
                'sample_size': config['n'],
                'confounding_strength': config['confounding'],
                'dataset_name': config['name']
            })

        all_results.extend(dataset_results)

    # Compile results
    df_results = pd.DataFrame(all_results)

    # Summary statistics
    print("\\n=== Summary Results ===")
    summary = df_results[df_results['success']].groupby(['algorithm', 'confounding_strength']).agg({
        'width': ['mean', 'std'],
        'computation_time': ['mean', 'std']
    }).round(4)

    print(summary)

    # Best performing algorithm by scenario
    print("\\n=== Best Algorithm by Scenario (narrowest bounds) ===")
    best_by_scenario = df_results[df_results['success']].loc[
        df_results[df_results['success']].groupby(['dataset_id'])['width'].idxmin()
    ][['dataset_name', 'algorithm', 'width']]

    print(best_by_scenario.to_string(index=False))

Specialized Use Cases
---------------------

Example 9: Custom Algorithm Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using a custom algorithm with the framework:

.. code-block:: python

   import numpy as np
   from causalboundingengine.algorithms.algorithm import Algorithm
   from causalboundingengine.scenarios import BinaryConf
   
   class ConservativeBounds(Algorithm):
       
       def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, 
                       conservatism: float = 0.8, **kwargs) -> tuple[float, float]:
           
           # Basic observed difference
           p1 = np.mean(Y[X == 1]) if np.any(X == 1) else 0.5
           p0 = np.mean(Y[X == 0]) if np.any(X == 0) else 0.5
           observed_diff = p1 - p0
           
           # Add conservative margin based on parameter
           margin = conservatism * (1 - abs(observed_diff))
           
           lower = observed_diff - margin
           upper = observed_diff + margin
           
           # Ensure bounds are valid
           lower = max(lower, -1.0)
           upper = min(upper, 1.0)
           
           return float(lower), float(upper)
   
   # Create custom scenario with new algorithm
   class CustomBinaryConf(BinaryConf):
       AVAILABLE_ALGORITHMS = {
           **BinaryConf.AVAILABLE_ALGORITHMS,
           'ATE': {
               **BinaryConf.AVAILABLE_ALGORITHMS['ATE'],
               'conservative': ConservativeBounds,
           }
       }
   
   # Use custom scenario
   X = np.array([0, 1, 1, 0, 1])
   Y = np.array([1, 0, 1, 0, 1])
   scenario = CustomBinaryConf(X, Y)
   
   print("=== Custom Algorithm Example ===")
   print(f"Standard Manski: {scenario.ATE.manski()}")
   print(f"Conservative (0.8): {scenario.ATE.conservative(conservatism=0.8)}")
   print(f"Conservative (0.3): {scenario.ATE.conservative(conservatism=0.3)}")

Example 10: Handling External Dependencies Gracefully
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robust code that handles missing R/Java dependencies:

.. code-block:: python

   import numpy as np
   from causalboundingengine.scenarios import BinaryConf, BinaryIV
   
   def robust_analysis(X, Y, Z=None, prefer_external=True):
       
       if Z is None:
           scenario = BinaryConf(X, Y)
           available_algorithms = scenario.get_algorithms('ATE')
       else:
           scenario = BinaryIV(X, Y, Z)
           available_algorithms = scenario.get_algorithms('ATE')
       
       results = {}
       
       # Priority order: external algorithms first if preferred
       if prefer_external:
           algorithm_priority = ['causaloptim', 'zaffalonbounds', 'autobound', 'manski']
       else:
           algorithm_priority = ['manski', 'autobound', 'causaloptim', 'zaffalonbounds']
       
       for alg_name in algorithm_priority:
           if alg_name in available_algorithms:
               try:
                   alg_func = getattr(scenario.ATE, alg_name)
                   bounds = alg_func()
                   results[alg_name] = {
                       'bounds': bounds,
                       'status': 'success',
                       'error': None
                   }
                   print(f"✓ {alg_name}: {bounds}")
               except ImportError as e:
                   results[alg_name] = {
                       'bounds': None,
                       'status': 'dependency_missing',
                       'error': str(e)
                   }
                   print(f"✗ {alg_name}: Missing dependency - {e}")
               except Exception as e:
                   results[alg_name] = {
                       'bounds': None,
                       'status': 'failed',
                       'error': str(e)
                   }
                   print(f"✗ {alg_name}: Failed - {e}")
       
       return results
   
   # Test with confounded data
   np.random.seed(42)
   X = np.random.binomial(1, 0.4, 100)
   Y = np.random.binomial(1, 0.3 + 0.2 * X, 100)
   
   print("=== Robust Analysis Example ===")
   print("Confounded scenario:")
   results_conf = robust_analysis(X, Y, prefer_external=True)
   
   # Test with IV data
   Z = np.random.binomial(1, 0.5, 100)
   print("\\nIV scenario:")
   results_iv = robust_analysis(X, Y, Z, prefer_external=True)
   
   # Summary
   successful_algorithms = [alg for alg, result in results_conf.items() 
                           if result['status'] == 'success']
   print(f"\\nSuccessful algorithms: {successful_algorithms}")

These examples demonstrate the flexibility and power of CausalBoundingEngine across various scenarios, from basic usage to advanced applications.
