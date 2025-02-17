import numpy as np
from scipy.special import softmax, kl_div
import warnings

_OFFSET = 1e-12

# Normalize voting scores to probabilities
def normalize(scores):
    # normalize between 0 and 1; input is a list of votes that may be negative
    votes = np.array(scores)
    votes = votes - np.min(votes)
    if np.sum(votes) == 0:
        return np.ones_like(votes) / len(votes)
    return votes / np.sum(votes)

def softmax(scores):
    # softmax function
    return np.exp(scores) / np.sum(np.exp(scores))

# Entropy calculation
def entropy(probs):
    # probs = [p1, p2, ..., pn], where p1 + p2 + ... + pn = 1
    return -np.sum(probs * np.log(probs + _OFFSET))

def joint_probs(probs_1, probs_2):
    """
    Calculate joint probabilities from marginal probabilities
    probs_1 and probs_2 are a 2d array of shape (n, m), where n is the 
    number of samples and m is the number of classes
    """
    assert probs_1.shape[0] == probs_2.shape[0]
    joint = np.zeros((probs_1.shape[0], probs_1.shape[1], probs_2.shape[1]))
    for i in range(probs_1.shape[0]):
        joint[i] = np.outer(probs_1[i], probs_2[i])
    return joint

def calculate_information(L:int, C:int) -> float:
    """
    Calculate the information of a system with L literals (going into the system) and C clauses.
    """
    return 1/(L*C)*np.log(1/(L*C))