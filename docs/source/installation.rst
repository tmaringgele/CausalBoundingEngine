Installation
============

CausalBoundingEngine can be installed via pip with optional dependencies for extended functionality.

Basic Installation
------------------

Install the core package:

.. code-block:: bash

   pip install causalboundingengine

This provides access to most algorithms including:

- Manski bounds
- Tian-Pearl bounds  
- Autobound
- EntropyBounds

Optional Dependencies
---------------------

The package supports additional functionality through optional extras:

R Support
~~~~~~~~~

For R-based algorithms like Causaloptim:

.. code-block:: bash

   pip install causalboundingengine[r]

**Requirements:**
   
- R must be installed and accessible in your system's PATH
- The ``causaloptim`` R package will be installed automatically via rpy2

**Manual R Setup:**

.. code-block:: bash

   # Install R (Ubuntu/Debian)
   sudo apt install r-base
   
   # Install R (macOS with Homebrew)
   brew install r
   
   # Install R (Windows)
   # Download from https://cran.r-project.org/

Java Support
~~~~~~~~~~~~

For Java-based algorithms like Zaffalonbounds:

.. code-block:: bash

   pip install causalboundingengine[java]

**Requirements:**

- Java 8+ must be installed
- JPype1 handles the Java Virtual Machine integration

**Manual Java Setup:**

.. code-block:: bash

   # Install Java (Ubuntu/Debian)
   sudo apt install default-jre default-jdk
   
   # Install Java (macOS with Homebrew)
   brew install openjdk
   
   # Install Java (Windows)
   # Download from https://adoptium.net/

Full Installation
~~~~~~~~~~~~~~~~~

Install all optional features:

.. code-block:: bash

   pip install causalboundingengine[full]

This is equivalent to:

.. code-block:: bash

   pip install causalboundingengine[r,java]

Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

For building documentation locally:

.. code-block:: bash

   pip install causalboundingengine[docs]

Development Installation
------------------------

For development and contributing:

.. code-block:: bash

   git clone https://github.com/yourusername/CausalBoundingEngine.git
   cd CausalBoundingEngine
   pip install -e .[full,docs]

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import causalboundingengine
   from causalboundingengine.scenarios import BinaryConf
   import numpy as np
   
   # Test basic functionality
   X = np.array([0, 1, 1, 0])
   Y = np.array([1, 0, 1, 1])
   scenario = BinaryConf(X, Y)
   bounds = scenario.ATE.manski()
   print(f"ATE bounds: {bounds}")

Test R integration (if installed):

.. code-block:: python

   # Test R-based algorithm
   try:
       bounds = scenario.ATE.causaloptim()
       print(f"Causaloptim bounds: {bounds}")
   except ImportError:
       print("R support not available")

Test Java integration (if installed):

.. code-block:: python

   # Test Java-based algorithm
   try:
       bounds = scenario.ATE.zaffalonbounds()
       print(f"Zaffalonbounds: {bounds}")
   except ImportError:
       print("Java support not available")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**R not found:**

.. code-block:: bash

   # Set R_HOME environment variable
   export R_HOME=/usr/lib/R  # Linux
   export R_HOME=/Library/Frameworks/R.framework/Resources  # macOS

**Java not found:**

.. code-block:: bash

   # Set JAVA_HOME environment variable
   export JAVA_HOME=/usr/lib/jvm/default-java  # Linux
   export JAVA_HOME=$(/usr/libexec/java_home)  # macOS

**Permission errors on Windows:**

Run commands in an elevated PowerShell/Command Prompt, or use:

.. code-block:: bash

   pip install --user causalboundingengine[full]

**Memory issues with Java algorithms:**

Increase JVM memory if needed:

.. code-block:: python

   import jpype
   jpype.startJVM("-Xmx4g")  # 4GB heap size

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~

**Windows:**
   - Use Anaconda/Miniconda for easier R and Java installation
   - May need Visual C++ redistributables for some dependencies

**macOS:**
   - Use Homebrew for R and Java installation
   - May need Xcode Command Line Tools

**Linux:**
   - Use package manager for R and Java
   - Ensure development headers are installed (``r-base-dev``, ``default-jdk``)

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

.. code-block:: text

   pandas >= 1.0.0
   numpy >= 1.18.0
   cvxpy >= 1.1.0
   pulp >= 2.0
   highspy >= 1.0.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   rpy2 >= 3.4.0          # R integration
   jpype1 >= 1.3.0        # Java integration
   sphinx >= 4.0.0        # Documentation
   sphinx_rtd_theme        # Documentation theme
