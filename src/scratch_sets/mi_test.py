import numpy as np
from scipy.stats import entropy
from typing import List, Tuple
from stats import joint_probs


def calculate_i(L:int, C:int, r:float)->float:

    return (C/L)#*(r+(C/L))

if __name__ == "__main__":
    # tests
    r = 1

    C_s = C_d = 200; C_t = 1600
    L_s, L_t, L_d = 784, 784, C_t*10
    I_s = calculate_i(L_s, C_s, r)
    I_t = calculate_i(L_t, C_t, r)
    I_d = calculate_i(L_d, C_d, r)
    print(f"I_s: {I_s:.4f}")
    print(f"I_t: {I_t:.4f}")
    print(f"I_d: {I_d:.4f}")