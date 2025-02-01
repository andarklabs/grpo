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

def appx_kl_div(p: NDArray[float], q: NDArray[float] ) -> float:
    """
    given two probability distributions, compute the average difference.

    wiki:
    http://joschu.net/blog/kl-approx.html
    """
    return np.sum(p * np.log(p /  q))

if __name__ == "__main__":
    # ground-truth distribution
    p = np.array([.2,.3,.5])
    # predicted distribution
    q = np.array([.2,.2,.6])
    print(kl_div(p, q))