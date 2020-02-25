import numpy as np
import sys

def oneStepLookaheadValue(trans_probs, reward, gamma):
    '''

    :param tp: SxAxS
    :param reward: S,
    :return: S,
    '''
    ns, na,_ = trans_probs.shape
    value = np.zeros(ns,)
    max_val = 0.
    for state in range(ns):
        for action in range(na):
            tp_from_s = trans_probs[state, action,:]
            curr_val = reward + gamma * np.dot(tp_from_s, value)
            if curr_val > max_val:
                max_val = curr_val
        value[state] = max_val
    return value

def valueIterationTillConvergence(transition_probabilities, reward,
                  gamma, convergence_threshold=0.01):

    diff = sys.float_info.max
    s,a,_ = transition_probabilities.shape
    old_val = np.zeros((s,))
    while diff > convergence_threshold*s:
        diff = 0
        new_values = oneStepLookaheadValue(transition_probabilities, reward, gamma)
        diff = np.sum(np.abs([new_values - old_val]))
        old_val = new_values
    return new_values

def getDeterministicPolicy(transition_probabilities, reward, v,
                  gamma):
    s,a,_ = transition_probabilities.shape

    if v is None:
        v = oneStepLookaheadValue(transition_probabilities, reward, gamma)

    policy = np.zeros((s,),dtype=np.int32)
    for state in range(s):
        exp_values=[]
        for action in range(a):
            exp_values.append(np.dot(transition_probabilities[s,a,:], reward + gamma*v))
        policy[state] = np.argmax(np.array(exp_values))

    return policy

def getStochasticPolicy(transition_probabilities, reward, v,
                  gamma):
    s,a,_ = transition_probabilities.shape

    if v is None:
        v = oneStepLookaheadValue(transition_probabilities, reward, gamma)

    Q = np.zeros(shape=[s,a])

    for state in range(s):
        for action in range(a):
            Q[state,action] = np.dot(transition_probabilities[state,action,:], reward + gamma*v)
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((s,1))
    return Q

def ValueIterationTable(tp, reward, gamma):
    '''

    :param tp: SxAxS
    :param reward: SxAxS
    :param gamma: float
    :return:
    '''
    value = valueIterationTillConvergence(tp, reward, gamma)
    #value: S
    vtable = np.zeros((tp.shape[0], tp.shape[1]))
    for state in range(tp.shape[0]):
        for action in range(tp.shape[1]):
            vtable[state,action] = tp[state, action] * (reward[state, action] + gamma*value)
    return vtable