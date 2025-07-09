import numpy as np
from causalboundingengine.algorithms.algorithm import Algorithm


class CausalOptim(Algorithm):
    def _compute_ATE(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray = None, **kwargs) -> tuple[float, float]:
        if Z is not None:
            return 0.2, 0.5
        else:
            return 0, 1.5
