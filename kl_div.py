from __future__ import annotations
import numpy as np
from numpy.typing import NDArray



def kl_div(p_r: NDArray[np.float32], p: NDArray[np.float32] ) -> float:
    """
    given two probability distributions, compute the average difference.

    wiki:
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    return np.sum(p_r * np.log(p_r /  p))

def unbiased_estimator_kl_div(p_r: NDArray[np.float32], p: NDArray[np.float32] ) -> NDArray[np.float32]:
    """
    unbiased estimator kl divergence (guaranteed to be positive)
    """
    return np.sum((p_r / p) - np.log(p_r / p) - 1)

if __name__ == "__main__":
    # ground-truth distribution (in our case: reference policy)
    p = np.array([.2,.3,.5])
    # predicted distribution
    q = np.array([.2,.2,.6])
    print(kl_div(p, q))
    print(unbiased_estimator_kl_div(p, q))