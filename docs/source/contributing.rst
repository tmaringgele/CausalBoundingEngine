Contributing
============

We welcome contributions to CausalBoundingEngine! This guide explains how to contribute effectively.

Getting Started
---------------

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork the Repository**

   Fork the repository on GitHub and clone your fork:

   .. code-block:: bash

      git clone https://github.com/yourusername/CausalBoundingEngine.git
      cd CausalBoundingEngine

2. **Create Development Environment**

   Using conda (recommended):

   .. code-block:: bash

      conda create -n causalbounding python=3.9
      conda activate causalbounding

   Or using venv:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\\Scripts\\activate

3. **Install Development Dependencies**

   .. code-block:: bash

      pip install -e .[full,docs]
      pip install pytest pytest-cov black isort mypy pre-commit

4. **Set Up Pre-commit Hooks**

   .. code-block:: bash

      pre-commit install

5. **Verify Installation**

   .. code-block:: bash

      python -c "import causalboundingengine; print('Success!')"
      pytest tests/ -v

Types of Contributions
----------------------

We welcome several types of contributions:

Algorithm Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

**What we need:**
   - New causal bounding algorithms
   - Implementations of existing algorithms from papers
   - Optimized versions of current algorithms

**Requirements:**
   - Inherit from ``Algorithm`` base class
   - Include comprehensive tests
   - Add documentation and examples
   - Cite original papers in docstrings

**Process:**
   1. Check existing issues for algorithm requests
   2. Create issue to discuss implementation
   3. Implement following the :doc:`extending` guide
   4. Submit pull request

Scenario Extensions
~~~~~~~~~~~~~~~~~~~

**What we need:**
   - New causal settings (e.g., multi-valued treatments)
   - Extensions to existing scenarios
   - Custom scenarios for specific domains

**Requirements:**
   - Extend ``Scenario`` base class
   - Define appropriate algorithm mappings
   - Include validation logic
   - Provide examples

Bug Fixes
~~~~~~~~~

**What we need:**
   - Fix algorithm computation errors
   - Resolve dependency issues
   - Improve error handling
   - Performance improvements

**Process:**
   1. Create issue describing the bug
   2. Include minimal reproducible example
   3. Fix and add regression test
   4. Submit pull request

Documentation
~~~~~~~~~~~~~

**What we need:**
   - Algorithm explanations
   - Usage examples
   - API documentation
   - Tutorials for specific domains

**Process:**
   1. Identify documentation gaps
   2. Write clear, comprehensive content
   3. Include code examples
   4. Build docs locally to verify

Testing
~~~~~~~

**What we need:**
   - Unit tests for new algorithms
   - Integration tests
   - Performance benchmarks
   - Edge case testing

Development Workflow
--------------------

Code Style
~~~~~~~~~~

We use several tools to maintain code quality:

**Black** for code formatting:

.. code-block:: bash

   black causalboundingengine/ tests/

**isort** for import sorting:

.. code-block:: bash

   isort causalboundingengine/ tests/

**mypy** for type checking:

.. code-block:: bash

   mypy causalboundingengine/

**flake8** for additional linting:

.. code-block:: bash

   flake8 causalboundingengine/ tests/

Run all checks:

.. code-block:: bash

   make lint  # If Makefile exists
   # Or manually:
   black --check causalboundingengine/ tests/
   isort --check causalboundingengine/ tests/
   mypy causalboundingengine/
   flake8 causalboundingengine/ tests/

Testing Guidelines
~~~~~~~~~~~~~~~~~~

**Test Structure:**

.. code-block:: python

   # tests/test_my_algorithm.py
   import numpy as np
   import pytest
   from causalboundingengine.algorithms.my_algorithm import MyAlgorithm

   class TestMyAlgorithm:
       
       def test_basic_functionality(self):
           \"\"\"Test basic algorithm functionality.\"\"\"
           X = np.array([0, 1, 1, 0])
           Y = np.array([1, 0, 1, 1])
           
           alg = MyAlgorithm()
           lower, upper = alg.bound_ATE(X, Y)
           
           assert isinstance(lower, float)
           assert isinstance(upper, float)
           assert lower <= upper
       
       def test_edge_cases(self):
           \"\"\"Test edge cases.\"\"\"
           # All same treatment
           X = np.array([1, 1, 1, 1])
           Y = np.array([0, 1, 0, 1])
           
           alg = MyAlgorithm()
           bounds = alg.bound_ATE(X, Y)
           assert not any(np.isnan(bounds))
       
       def test_parameter_validation(self):
           \"\"\"Test parameter validation.\"\"\"
           X = np.array([0, 1])
           Y = np.array([1, 0])
           
           alg = MyAlgorithm()
           
           # Valid parameter
           alg.bound_ATE(X, Y, param=0.5)
           
           # Invalid parameter
           with pytest.raises(ValueError):
               alg.bound_ATE(X, Y, param=-1.0)

**Run Tests:**

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run specific test file
   pytest tests/test_my_algorithm.py

   # Run with coverage
   pytest tests/ --cov=causalboundingengine --cov-report=html

   # Run specific test
   pytest tests/test_my_algorithm.py::TestMyAlgorithm::test_basic_functionality

Documentation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

**Docstring Format:**

Use NumPy-style docstrings:

.. code-block:: python

   def my_function(X: np.ndarray, Y: np.ndarray, param: float = 1.0) -> tuple[float, float]:
       \"\"\"
       Compute bounds using my method.
       
       This function implements the algorithm described in Author (2023).
       The method works by...
       
       Parameters
       ----------
       X : np.ndarray
           Binary treatment array of shape (n,) with values in {0, 1}.
       Y : np.ndarray
           Binary outcome array of shape (n,) with values in {0, 1}.
       param : float, default=1.0
           Algorithm parameter. Must be positive.
           
       Returns
       -------
       tuple[float, float]
           Lower and upper bounds on the causal effect.
           
       Raises
       ------
       ValueError
           If param is not positive.
           
       Notes
       -----
       The algorithm assumes that...
       
       References
       ----------
       Author, A. (2023). Important Paper. Journal Name, 1(1), 1-10.
       
       Examples
       --------
       >>> import numpy as np
       >>> X = np.array([0, 1, 1, 0])
       >>> Y = np.array([1, 0, 1, 1])
       >>> bounds = my_function(X, Y, param=2.0)
       >>> print(bounds)
       (0.1, 0.9)
       \"\"\"

**Building Documentation:**

.. code-block:: bash

   cd docs/
   make html
   # Open docs/build/html/index.html

Contribution Process
--------------------

Step-by-Step Guide
~~~~~~~~~~~~~~~~~~

1. **Create Issue (Optional but Recommended)**

   Before starting work, create an issue to discuss your contribution:

   .. code-block:: text

      Title: Add [Algorithm Name] algorithm
      
      Description:
      I would like to implement the [Algorithm Name] algorithm from [Paper Citation].
      
      This algorithm:
      - Addresses [specific causal setting]
      - Provides [type of bounds]
      - Has advantages: [list advantages]
      
      Implementation plan:
      - [ ] Core algorithm implementation
      - [ ] Unit tests
      - [ ] Integration with BinaryConf scenario
      - [ ] Documentation and examples

2. **Create Feature Branch**

   .. code-block:: bash

      git checkout -b feature/my-new-algorithm
      # or
      git checkout -b fix/issue-123

3. **Implement Changes**

   Follow the development guidelines and implement your changes.

4. **Add Tests**

   Ensure your code is well-tested:

   .. code-block:: bash

      pytest tests/test_my_new_feature.py -v

5. **Update Documentation**

   Add or update relevant documentation files.

6. **Run Quality Checks**

   .. code-block:: bash

      black causalboundingengine/ tests/
      isort causalboundingengine/ tests/
      mypy causalboundingengine/
      pytest tests/ --cov=causalboundingengine

7. **Commit Changes**

   Use descriptive commit messages:

   .. code-block:: bash

      git add .
      git commit -m "Add MyAlgorithm implementation
      
      - Implement core algorithm logic
      - Add comprehensive unit tests  
      - Update BinaryConf scenario
      - Add documentation and examples
      
      Fixes #123"

8. **Push and Create Pull Request**

   .. code-block:: bash

      git push origin feature/my-new-algorithm

   Then create a pull request on GitHub.

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

**PR Title Format:**

.. code-block:: text

   [TYPE] Brief description
   
   Examples:
   [FEAT] Add Bayesian bounds algorithm
   [FIX] Resolve memory leak in AutoBound
   [DOCS] Update installation instructions
   [TEST] Add integration tests for IV scenarios

**PR Description Template:**

.. code-block:: text

   ## Description
   Brief description of changes and motivation.
   
   ## Type of Change
   - [ ] Bug fix (non-breaking change that fixes an issue)
   - [ ] New feature (non-breaking change that adds functionality)
   - [ ] Breaking change (fix or feature that changes existing functionality)
   - [ ] Documentation update
   
   ## Testing
   - [ ] Added unit tests
   - [ ] Added integration tests
   - [ ] All tests pass locally
   - [ ] Tested on multiple Python versions
   
   ## Documentation
   - [ ] Updated docstrings
   - [ ] Updated user guide
   - [ ] Added examples
   - [ ] Built docs successfully
   
   ## Checklist
   - [ ] Code follows style guidelines (black, isort, mypy)
   - [ ] Self-review completed
   - [ ] Appropriate reviewers assigned
   - [ ] Related issues linked

Review Process
~~~~~~~~~~~~~~

**What Reviewers Look For:**

1. **Correctness**: Does the implementation correctly solve the problem?
2. **Testing**: Are there comprehensive tests covering edge cases?
3. **Documentation**: Is the code well-documented and examples clear?
4. **Style**: Does the code follow project conventions?
5. **Performance**: Are there any obvious performance issues?
6. **API Design**: Is the interface consistent with existing code?

**Addressing Review Comments:**

.. code-block:: bash

   # Make requested changes
   git add .
   git commit -m "Address review comments
   
   - Fix edge case handling
   - Add missing type hints
   - Improve error messages"
   
   git push origin feature/my-new-algorithm

**After Approval:**

Maintainers will merge your PR. You can then:

.. code-block:: bash

   git checkout main
   git pull upstream main
   git branch -d feature/my-new-algorithm

Specific Contribution Areas
---------------------------

Priority Algorithms
~~~~~~~~~~~~~~~~~~~

We're particularly interested in implementations of:

1. **Balke-Pearl bounds** for IV settings
2. **Robins bounds** for time-varying treatments  
3. **Sensitivity analysis methods** (e.g., E-values)
4. **Machine learning enhanced bounds**
5. **Bounds for continuous outcomes**

External Integration
~~~~~~~~~~~~~~~~~~~

Help improve integration with:

1. **R packages**: Better rpy2 integration
2. **Julia packages**: Add Julia backend support  
3. **Stan/PyMC**: Bayesian bounding approaches
4. **Optimization libraries**: Better solver interfaces

Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~

Areas needing optimization:

1. **Large dataset handling**: Memory-efficient algorithms
2. **Parallel computation**: Multi-core algorithm implementations
3. **Approximate algorithms**: Fast approximations for exploration
4. **Caching**: Intelligent result caching

Getting Help
------------

Where to Get Help
~~~~~~~~~~~~~~~~~

1. **GitHub Discussions**: General questions and ideas
2. **GitHub Issues**: Bug reports and feature requests
3. **Documentation**: Check existing docs first
4. **Code Review**: Ask for feedback during PR process

Communication Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

- Be respectful and constructive
- Provide minimal reproducible examples for bugs
- Search existing issues before creating new ones
- Use descriptive titles and clear descriptions
- Follow up on your contributions

Recognition
-----------

Contributors are recognized in:

- **AUTHORS.md**: All contributors listed
- **CHANGELOG**: Major contributions noted in releases
- **Documentation**: Algorithm implementers credited in docs
- **Citations**: Academic contributors cited in papers

Thank you for contributing to CausalBoundingEngine! Your efforts help make causal inference more accessible to researchers and practitioners worldwide.
