import numpy as np
import cvxopt
from typing import List, Tuple, Dict

def IRL(transition_probs: np.array,
             pi: np.array, gamma: float, Rmax: float, l1: float)-> np.array:
    '''
    :param transition_probs: SxAxS
    :param pi: shape ->(S,) deterministic policy
    :param gamma: discount
    :param Rmax: upper bound on reward
    :param l1: regularization
    :return: reward array
    '''
    num_states, num_actions,_ = transition_probs.shape
    A = set(range(num_actions))


    ones_s = np.ones(num_states)
    zero_s = np.zeros(num_states)
    zero_ss = np.zeros((num_states, num_states))

    tp = np.transpose(transition_probs, [1,0,2])

    def I(*args):
        return np.eye(*args)

    #curry the function to avoid passing around so much stuff
    def objective(state, action):
        return np.dot(tp[pi[state], state] - tp[action, state], np.linalg.inv(I(num_states) - gamma*tp[pi[state]]))

    raise NotImplementedError


