from __future__ import annotations
import numpy as np
from numpy.typing import NDArray



def kl_div(p: NDArray[float], q: NDArray[float] ) -> float:
    """
    given two probability distributions, compute the average difference.

    wiki:
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    return np.sum(p * np.log(p /  q))

def unbiased_estimator_kl_div(p: NDArray[float], q: NDArray[float] ) -> NDArrray[float]:
    """
    unbiased estimator kl divergence (guaranteed to be positive)
    """
    return (p / q) - np.log(p / q) - 1

if __name__ == "__main__":
    # ground-truth distribution (in our case: reference policy)
    p = np.array([.2,.3,.5])
    # predicted distribution
    q = np.array([.2,.2,.6])
    print(kl_div(p, q))
    print(unbiased_estimator_kl_div(p, q))