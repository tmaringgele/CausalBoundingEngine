

import numpy as np
from causalboundingengine.utils.alg_util import AlgUtil


class Manski:

    def bound_ATE(X, Y):
        """
        Compute Manski bounds for the the using only observed treatment (X) and outcome (Y).
        Args:
            X (np.ndarray): Treatment assignment (1 for treated, 0 for control).
            Y (np.ndarray): Observed outcomes.
        Returns:
            tuple: Lower and upper bounds for the Average Treatment Effect (ATE) as numpy floats.
        """

        failed = False

        try:
            p1 = np.mean(Y[X == 1]) if np.any(X == 1) else 0.0
            p0 = np.mean(Y[X == 0]) if np.any(X == 0) else 0.0

            lower = p1 - p0 - 1
            upper = p1 - p0 + 1
            lower = max(lower, -1)
            upper = min(upper, 1)

            # Ensure logical ordering
            lower, upper = min(lower, upper), max(lower, upper)

        except Exception:
            failed = True

        # Flatten bounds to trivial ceils
        lower, upper = AlgUtil.flatten_bounds_to_trivial_ceils('ATE', lower, upper, failed)


        return lower, upper
        
