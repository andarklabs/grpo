import numpy as np
import kl_div as kl

EPSILON = 0
BETA = 0
G = 6

def main(o_i, p_r, p_o, p, a, b = BETA, g = G):
    for o_i in range(g):
        a = advantage(o_i)
        return np.sum(np.sum(min(p,p_o,a)-b*kl.unbiased_estimator_kl_div(p_r,p))/len(o_i))/g
    # print("Hello from grpo!")


def advantage(o_i):
    pass

def min(p,p_o,a, e = EPSILON):
    return np.min(p/p_o, np.clip(p/p_o,1-e,1+e))*a

'''def cycle(o_i, p_r, p_o, p, a, b = BETA):
    return np.sum(min(p,p_o,a)-b*kl.unbiased_estimator_kl_div(p_r,p))/len(o_i)'''

if __name__ == "__main__":
    main()
