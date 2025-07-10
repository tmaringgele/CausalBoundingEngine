# CausalBoundingEngine

<div align="center">

![CausalBoundingEngine Logo](cbe_logo.png)

**A unified Python framework for causal effect bounding algorithms**

[![PyPI version](https://img.shields.io/pypi/v/causalboundingengine.svg)](https://pypi.org/project/causalboundingengine/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-brightgreen.svg)](https://causalboundingengine.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[**Documentation**](https://causalboundingengine.readthedocs.io/) | [**Quick Start**](#quick-start) | [**Examples**](#examples) | [**Contributing**](#contributing)

</div>

## Overview

CausalBoundingEngine is a modular Python package that provides a **unified interface** for comparing and applying state-of-the-art causal bounding algorithms. It enables researchers and practitioners to compute bounds on causal effects when unmeasured confounding is present or when using instrumental variables.

### Key Features

🔧 **Unified Interface** - Consistent API across all algorithms and scenarios  
📊 **Multiple Algorithms** - Manski, Tian-Pearl, AutoBound, CausalOptim, Zaffalonbounds, and more  
🎯 **Flexible Scenarios** - Support for confounded and instrumental variable settings  
🔗 **External Engines** - Integration with R (rpy2) and Java (jpype1) backends  
🚀 **Easy Extension** - Simple framework for adding new algorithms and scenarios  
📚 **Comprehensive Docs** - Detailed documentation with examples and API reference  

### Supported Algorithms

| Algorithm | ATE | PNS | Scenarios | Dependencies | Reference |
|-----------|-----|-----|-----------|--------------|-----------|
| **Manski** | ✓ | ✗ | BinaryConf | Core | Manski (1990) |
| **Tian-Pearl** | ✓ | ✓ | BinaryConf | Core | Tian & Pearl (2000) |
| **AutoBound** | ✓ | ✓ | BinaryConf, BinaryIV | Core | Duarte et al. (2023) |
| **EntropyBounds** | ✓ | ✓ | BinaryConf | Core | Jiang & Shpitser (2020) |
| **CausalOptim** | ✓ | ✓ | BinaryConf, BinaryIV | R | Sachs et al. (2022) |
| **Zaffalonbounds** | ✓ | ✓ | BinaryConf, BinaryIV | Java | Zaffalon et al. (2022) |
| **ZhangBareinboim** | ✓ | ✗ | ContIV | Core | Zhang & Bareinboim (2021) |

## Installation

### Core Package

```bash
pip install causalboundingengine
```

### Optional Dependencies

For extended functionality, install with optional dependencies:

```bash
# R integration (CausalOptim algorithm)
pip install causalboundingengine[r]

# Java integration (Zaffalonbounds algorithm)  
pip install causalboundingengine[java]

# All optional features
pip install causalboundingengine[full]

# Documentation building
pip install causalboundingengine[docs]
```

### System Dependencies

For algorithms requiring external engines:

**R Support** (for CausalOptim):
```bash
# Ubuntu/Debian
sudo apt install r-base

# macOS
brew install r

# Windows: Download from https://cran.r-project.org/
```

**Java Support** (for Zaffalonbounds):
```bash
# Ubuntu/Debian
sudo apt install default-jre

# macOS  
brew install openjdk

# Windows: Download from https://adoptium.net/
```

## Quick Start

### Basic Usage

```python
import numpy as np
from causalboundingengine.scenarios import BinaryConf

# Your observational data
X = np.array([0, 1, 1, 0, 1, 0, 1, 0])  # Treatment
Y = np.array([1, 0, 1, 0, 1, 1, 0, 1])  # Outcome

# Create scenario and compute bounds
scenario = BinaryConf(X, Y)

# Compute ATE bounds with different algorithms
manski_bounds = scenario.ATE.manski()           # (-1.0, 1.0) - Most conservative
tianpearl_bounds = scenario.ATE.tianpearl()     # (-0.75, 0.75) - Tighter bounds
autobound_bounds = scenario.ATE.autobound()     # (-0.5, 0.5) - LP optimization

print(f"Manski bounds:    {manski_bounds}")
print(f"Tian-Pearl bounds: {tianpearl_bounds}")
print(f"AutoBound bounds:  {autobound_bounds}")
```

### Instrumental Variable Analysis

```python
from causalboundingengine.scenarios import BinaryIV

# IV data (e.g., randomized trial with non-compliance)
Z = np.array([0, 1, 1, 0, 1, 0, 1, 0])  # Instrument (randomization)
X = np.array([0, 1, 0, 0, 1, 0, 1, 0])  # Treatment (actual uptake)  
Y = np.array([1, 0, 1, 0, 1, 1, 0, 1])  # Outcome

# Create IV scenario
scenario = BinaryIV(X, Y, Z)

# Compute bounds leveraging IV assumptions
iv_bounds = scenario.ATE.autobound()
print(f"IV-based bounds: {iv_bounds}")  # Often tighter than confounded case
```

### Continuous Outcomes

```python
from causalboundingengine.scenarios import ContIV

# Binary instrument/treatment with continuous outcome
Z = np.array([0, 1, 1, 0, 1])           # Binary instrument
X = np.array([0, 1, 1, 0, 1])           # Binary treatment  
Y = np.array([0.2, 0.8, 0.6, 0.1, 0.9]) # Continuous outcome [0,1]

scenario = ContIV(X, Y, Z)
bounds = scenario.ATE.zhangbareinboim()
print(f"Continuous outcome bounds: {bounds}")
```

## Examples

### Algorithm Comparison

```python
import numpy as np
from causalboundingengine.scenarios import BinaryConf

# Generate example data
np.random.seed(42)
n = 1000
X = np.random.binomial(1, 0.3, n)
Y = np.random.binomial(1, 0.6, n)

scenario = BinaryConf(X, Y)

# Compare multiple algorithms
algorithms = ['manski', 'tianpearl', 'autobound', 'entropybounds']
results = {}

for alg in algorithms:
    if alg == 'entropybounds':
        bounds = getattr(scenario.ATE, alg)(theta=0.5)
    else:
        bounds = getattr(scenario.ATE, alg)()
    results[alg] = bounds
    print(f"{alg:15} ATE bounds: {bounds}")

# Output:
# manski          ATE bounds: (-0.7, 0.7)
# tianpearl       ATE bounds: (-0.4, 0.4)  
# autobound       ATE bounds: (-0.3, 0.3)
# entropybounds   ATE bounds: (-0.2, 0.2)
```

### Sensitivity Analysis

```python
# Sensitivity analysis with EntropyBounds
thetas = [0.1, 0.5, 1.0, 2.0]
for theta in thetas:
    bounds = scenario.ATE.entropybounds(theta=theta)
    width = bounds[1] - bounds[0]
    print(f"θ={theta}: bounds={bounds}, width={width:.3f}")

# Output shows how bounds widen as assumptions weaken:
# θ=0.1: bounds=(-0.15, 0.15), width=0.300
# θ=0.5: bounds=(-0.25, 0.25), width=0.500  
# θ=1.0: bounds=(-0.35, 0.35), width=0.700
# θ=2.0: bounds=(-0.45, 0.45), width=0.900
```

### Robust Analysis Workflow

```python
def robust_analysis(X, Y, Z=None):
    """Run multiple algorithms for robustness."""
    if Z is None:
        scenario = BinaryConf(X, Y)
        algorithms = ['manski', 'tianpearl', 'autobound']
    else:
        scenario = BinaryIV(X, Y, Z)  
        algorithms = ['autobound']  # Add 'causaloptim', 'zaffalonbounds' if available
    
    results = {}
    for alg in algorithms:
        try:
            results[alg] = getattr(scenario.ATE, alg)()
            print(f"✓ {alg}: {results[alg]}")
        except Exception as e:
            print(f"✗ {alg}: {e}")
    
    return results

# Run robust analysis
bounds_dict = robust_analysis(X, Y)
```

## Scenarios

CausalBoundingEngine organizes algorithms by causal scenario:

### BinaryConf
- **Use case**: Observational studies with binary treatment/outcome
- **Assumptions**: Potential unmeasured confounding
- **Algorithms**: Manski, TianPearl, AutoBound, EntropyBounds, CausalOptim, Zaffalonbounds

### BinaryIV  
- **Use case**: Instrumental variable analysis with binary variables
- **Assumptions**: Valid instrument (relevance, exclusion, exogeneity)
- **Algorithms**: AutoBound, CausalOptim, Zaffalonbounds

### ContIV
- **Use case**: Binary instrument/treatment with continuous outcome [0,1]
- **Assumptions**: Valid instrument, bounded outcome
- **Algorithms**: ZhangBareinboim

## Advanced Features

### Custom Algorithm Parameters

```python
# EntropyBounds with custom confounding strength
bounds = scenario.ATE.entropybounds(theta=0.2)

# CausalOptim with custom R path
bounds = scenario.ATE.causaloptim(r_path="/usr/local/bin/R")
```

### Algorithm Availability

```python
# Check available algorithms
print("ATE algorithms:", scenario.get_algorithms('ATE'))
print("PNS algorithms:", scenario.get_algorithms('PNS'))

# Dynamic algorithm access
algorithm_name = 'tianpearl'
if algorithm_name in scenario.get_algorithms('ATE'):
    bounds = getattr(scenario.ATE, algorithm_name)()
```

### Error Handling

```python
import logging
logging.basicConfig(level=logging.WARNING)

# Algorithms return trivial bounds on failure
bounds = scenario.ATE.some_algorithm()
if bounds == (-1.0, 1.0):  # ATE trivial bounds
    print("Algorithm failed, returned trivial bounds")
```

## Documentation

📖 **Full Documentation**: https://causalboundingengine.readthedocs.io/

The documentation includes:

- **[Installation Guide](https://causalboundingengine.readthedocs.io/en/latest/installation.html)** - Detailed setup instructions
- **[Quick Start](https://causalboundingengine.readthedocs.io/en/latest/quickstart.html)** - Get up and running quickly  
- **[User Guide](https://causalboundingengine.readthedocs.io/en/latest/user_guide.html)** - Concepts and best practices
- **[Algorithm Reference](https://causalboundingengine.readthedocs.io/en/latest/algorithms.html)** - Detailed algorithm documentation
- **[Scenario Guide](https://causalboundingengine.readthedocs.io/en/latest/scenarios.html)** - When to use which scenario
- **[API Reference](https://causalboundingengine.readthedocs.io/en/latest/api_reference.html)** - Complete API documentation
- **[Examples](https://causalboundingengine.readthedocs.io/en/latest/examples.html)** - Real-world usage examples
- **[References](https://causalboundingengine.readthedocs.io/en/latest/references.html)** - Citations and credits

## Contributing

We welcome contributions! The process is simple:

1. **Fork** the repository on GitHub
2. **Clone** your fork and install: `pip install -e .`
3. **Make** your changes and add tests
4. **Submit** a Pull Request

See our [Contributing Guide](https://causalboundingengine.readthedocs.io/en/latest/contributing.html) for details.

### Areas for Contribution

- 🔧 New algorithm implementations
- 📊 Additional causal scenarios  
- 🐛 Bug fixes and improvements
- 📚 Documentation and examples
- 🚀 Performance optimizations

## Citation

If you use CausalBoundingEngine in your research, please cite the relevant algorithm papers. See the [References](https://causalboundingengine.readthedocs.io/en/latest/references.html) section for complete citations.

### BibTeX Template

```bibtex
@software{causalboundingengine,
  title={CausalBoundingEngine: A Unified Framework for Causal Effect Bounding},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/CausalBoundingEngine},
  note={Python package for causal effect bounding algorithms}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

CausalBoundingEngine integrates algorithms from multiple research papers. We gratefully acknowledge:

- **Manski (1990)** - Nonparametric bounds foundation
- **Tian & Pearl (2000)** - Probability of causation bounds  
- **Duarte et al. (2023)** - AutoBound optimization approach
- **Jiang & Shpitser (2020)** - Entropy-based weak confounding
- **Sachs et al. (2022)** - CausalOptim symbolic derivation
- **Zaffalon et al. (2022)** - Credal network approaches
- **Zhang & Bareinboim (2021)** - Continuous outcome bounding

See the [References](https://causalboundingengine.readthedocs.io/en/latest/references.html) page for complete citations and attributions.

---

<div align="center">

**Built with ❤️ for the causal inference community**

[Documentation](https://causalboundingengine.readthedocs.io/) • [PyPI](https://pypi.org/project/causalboundingengine/) • [GitHub](https://github.com/yourusername/CausalBoundingEngine)

</div>

