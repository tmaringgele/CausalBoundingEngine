from abc import ABC, abstractmethod
import numpy as np
from causalboundingengine.utils.alg_util import AlgUtil

class Algorithm(ABC):
    def bound_ATE(self, *args, **kwargs) -> tuple[float, float]:
        failed = False

        try:
            lower, upper = self._compute_bounds(*args, **kwargs)
        except Exception:
            print("Error in computing bounds, returning trivial bounds.")
            ## print exception
            failed = True
            lower, upper = None, None

        lower, upper = AlgUtil.flatten_bounds_to_trivial_ceils('ATE', lower, upper, failed)
        return lower, upper

    @abstractmethod
    def _compute_bounds(self, *args, **kwargs) -> tuple[float, float]:
        pass