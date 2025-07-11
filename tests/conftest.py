import pytest
import numpy as np


@pytest.fixture
def sample_binary_data():
    """Fixture providing sample binary data for testing."""
    np.random.seed(42)  # For reproducible tests
    n = 50
    X = np.random.binomial(1, 0.3, n)
    Y = np.random.binomial(1, 0.6, n)
    return X, Y


@pytest.fixture
def sample_iv_data():
    """Fixture providing sample IV data for testing."""
    np.random.seed(42)
    n = 50
    Z = np.random.binomial(1, 0.5, n)  # Instrument
    X = np.random.binomial(1, 0.3 + 0.4 * Z, n)  # Treatment influenced by Z
    Y = np.random.binomial(1, 0.4 + 0.3 * X, n)  # Outcome influenced by X
    return Z, X, Y


@pytest.fixture
def sample_continuous_data():
    """Fixture providing sample continuous outcome data."""
    np.random.seed(42)
    n = 20
    Z = np.random.binomial(1, 0.5, n)
    X = np.random.binomial(1, 0.5, n)
    Y = np.random.uniform(0, 1, n)  # Continuous outcome in [0,1]
    return Z, X, Y


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "external: marks tests requiring external dependencies (R, Java)"
    )
