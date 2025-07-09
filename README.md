# CausalBoundingEngine

CausalBoundingEngine is a modular Python package for comparing and applying causal bounding algorithms.

## Features
- Multiple algorithms (Manski, Tian-Pearl, CausalOptim, etc.)
- Support for IV and confounded scenarios
- External engine support (R, Java)

## Installation

You can install the core package via pip:

```bash
pip install causalboundingengine
```

### Optional Features

The package supports additional functionality that relies on optional system or language dependencies:

| Extra      | Description                         | Install Command                                 |
|------------|-------------------------------------|-------------------------------------------------|
| `r`        | Support for R-based algorithms (e.g. `causaloptim`) | `pip install causalboundingengine[r]` |
| `java`     | Support for Java-based algorithms (e.g. via `jpype1`) | `pip install causalboundingengine[java]` |
| `full`     | All optional features (R + Java)    | `pip install causalboundingengine[full]`       |
| `docs`     | Build the documentation locally     | `pip install causalboundingengine[docs]`       |

> 💡 To use R-based algorithms like `causaloptim`, make sure:
> - R is installed and accessible in your system's PATH
> - Optionally set `R_HOME` manually or pass it via `r_path="..."` when calling the algorithm


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

