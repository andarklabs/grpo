from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def relative_advantage(r_i: float, r: NDArray[np.float32]) -> float:
    return (r_i  - np.mean(r) / np.std(r))

if __name__ == "__main__":
    r = np.array([.2,.3,.5])
    r_i = .2
    print(relative_advantage(r_i, r))