# CausalBoundingEngine

CausalBoundingEngine is a modular Python package for comparing and applying causal bounding algorithms.

## Features
- Multiple algorithms (Manski, Tian-Pearl, CausalOptim, etc.)
- Support for IV and confounded scenarios
- External engine support (R, Java)

## Installation
```bash
pip install causalboundingengine
```

### External Dependencies
- **R**: Required by `rpy2`
- **Java**: Required by `jpype1`

Install on Ubuntu:
```bash
sudo apt install r-base default-jre
```

## Usage
```python
from boundingengine.algorithms import manski
manski.bound(query, data)
```

