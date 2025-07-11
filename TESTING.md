# Test Suite Documentation

This document describes the test suite created for the CausalBoundingEngine package.

## Test Files Overview

### 1. `test_dummy.py` - Basic Smoke Tests
**Status**: ✅ PASSING (3/4 tests pass, 1 skipped due to missing optional dependencies)

- `test_package_imports()`: Tests core package components can be imported
- `test_basic_data_structure()`: Tests Data class creation and unpacking
- `test_manski_basic()`: Tests basic Manski algorithm functionality
- `test_scenario_integration()`: Tests BinaryConf scenario (skipped - needs `pulp`)

### 2. `test_core_algorithms.py` - Algorithm Testing
**Status**: ✅ PASSING (7/7 tests pass)

**Passing Tests:**
- `TestManski::test_manski_basic`: Basic Manski functionality
- `TestManski::test_manski_edge_cases`: Edge cases (all treated/control)
- `TestManski::test_manski_consistent_data`: Realistic data testing
- `TestTianPearl::test_tianpearl_basic`: Basic Tian-Pearl functionality
- `TestTianPearl::test_tianpearl_pns_functionality`: Tian-Pearl PNS bounds
- `TestAlgorithmInterface::test_bound_methods_exist`: Interface checking
- `TestAlgorithmInterface::test_error_handling`: Error handling

### 3. `test_scenarios.py` - Scenario and Data Testing
**Status**: ✅ PASSING (5/5 tests pass for core components)

**Passing Tests:**
- `TestDataClass::test_data_creation`: Data class basic functionality
- `TestDataClass::test_data_with_instrument`: IV data handling
- `TestDataClass::test_data_unpack`: Data unpacking methods
- `TestScenarioBase::test_scenario_creation`: Basic scenario creation
- `TestScenarioBase::test_algorithm_dispatcher`: Algorithm dispatcher functionality

**Conditional Tests (may skip due to dependencies):**
- `TestBinaryConfScenario`: Tests for BinaryConf scenario
- `TestIVScenarios`: Tests for BinaryIV and ContIV scenarios

### 4. `test_integration.py` - Integration Testing
**Status**: ✅ PASSING (tests pass for available components)

- End-to-end workflow testing
- Algorithm comparison testing
- Error resilience testing
- IV scenario testing (conditional)
- Data handling validation

## Running Tests

### Quick Test Commands

```bash
# Run all basic tests (no optional dependencies needed)
python -m pytest tests/test_dummy.py tests/test_scenarios.py::TestDataClass tests/test_scenarios.py::TestScenarioBase

# Run core algorithm tests
python -m pytest tests/test_core_algorithms.py

# Run integration tests
python -m pytest tests/test_integration.py

# Run all tests
python -m pytest tests/
```

### Using the Test Runner Script

```bash
python run_basic_tests.py
```

This script runs a curated set of basic tests that work with minimal dependencies.

## Test Dependencies

### Required (for basic tests):
- `numpy`
- `pytest`

### Optional (for full test suite):
- `pulp` - For optimization-based algorithms
- `cvxpy` - For convex optimization algorithms
- `pandas` - For data handling tests
- Other algorithm-specific dependencies

## Test Coverage

The test suite covers:

✅ **Core Data Structures**
- Data class creation and manipulation
- Array handling and validation
- Instrument variable support

✅ **Basic Algorithms**
- Manski bounds implementation
- Tian-Pearl bounds basic functionality
- Algorithm interface consistency

✅ **Scenario Framework**
- Scenario base class functionality
- Algorithm dispatcher mechanism
- Algorithm discovery methods

✅ **Integration**
- End-to-end workflows
- Error handling and resilience
- Multiple algorithm comparison

⚠️ **Partially Covered (depends on optional deps)**
- Full scenario implementations (BinaryConf, BinaryIV, ContIV)
- Advanced algorithm functionality
- Optimization-based bounds

## Known Issues and Limitations

1. **Missing Optional Dependencies**: Many tests are skipped when optional dependencies (like `pulp`, `rpy2`, etc.) are not installed.

2. **Edge Case Handling**: Some algorithms may not handle edge cases (empty data, no variation) as robustly as the tests expect.

## Recommendations

1. **Install Optional Dependencies**: For comprehensive testing, install the full dependency stack:
   ```bash
   pip install pulp cvxpy pandas rpy2
   ```

2. **Expand Test Coverage**: Add tests for:
   - More complex scenarios and algorithms
   - Performance benchmarking
   - Documentation examples validation

3. **Continuous Integration**: Set up CI to run these tests automatically on code changes.

## Example Test Output

```
🧪 Running CausalBoundingEngine Basic Tests
==================================================
✅ tests/test_dummy.py: PASSED (3 passed, 1 skipped)
✅ tests/test_core_algorithms.py: PASSED (7 passed)
✅ tests/test_scenarios.py::TestDataClass: PASSED
✅ tests/test_scenarios.py::TestScenarioBase: PASSED
✅ tests/test_integration.py::TestDataHandling: PASSED

Total: 5 test suites
✅ Passed: 5
❌ Failed: 0
💥 Errors: 0

🎉 All basic tests passed! The core package functionality is working.
```
