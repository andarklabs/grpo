import numpy as np
import kl_div as kl

EPSILON = 0
BETA = 0

def main():
    print("Hello from grpo!")
    return 


def min(p,p_o,a):
    return np.min(p/p_o, np.clip(p/p_o,1-EPSILON,1+EPSILON))*a

def cycle():
    pass

if __name__ == "__main__":
    main()
