import numpy as np
import random


'''YOU MAY CHOOSE TO REPLACE THE ASSERTIONS WITH try-catch BLOCKS.
    I PREFER USING ASSERTIONS

    P: new_state x state x action
    '''



def noise(shape, multiplier = 1.):
    return multiplier * np.random.randn(*shape)

def bellman(V_old, P, R):
    num_S, num_A = R.shape
    tiny = 1e-8
    Q_new = np.ones(shape=[num_S, num_A])

    for state in range(num_S):
        for action in range(num_A):
            Q_new[state, action] = R[state, action] + np.dot(P[state, action, :], V_old)
    Q_new = Q_new + noise([num_S, num_A], multiplier=tiny)
    V_new = np.max(Q_new, 1)
    pi_new = np.argmax(Q_new, 1)

    return V_new, pi_new


def sampleDiscrete(p,n):
    sum = np.sum(p)
    assert sum > 0.99 and sum < 1.01
    return np.argmax(np.random.multinomial(n,p))

def fixedHorizonValueIteration(P, R, num_horizons):
    '''

    :param P: SxAxS transition probs
    :param R: SxA rewards
    :param num_horizons: int
    :return:
    '''
    assert num_horizons > 0
    num_S, num_A = R.shape
    V_old = np.zeros([num_S,1])

    done = False
    for i in range(num_horizons):
        V_better, pi = bellman(V_old, P, R)
        V_old = V_better
    return V_better, pi #ignore the warning, already have an assertion

def infiniteHorizonValueIteration(P,R):
    raise NotImplementedError


def getOptimisticProbTransitions(V_old, pHat, dist):
    '''    :param V_old: Sx1 value fn.
    :param pHat: Sx1 current estimated transition probs
    :param dist: slack
    :return: Sx1 optimistic probability transitions
    '''
    num_S = pHat.shape[0]
    p_optimistic = pHat

    best_action = np.argmax(V_old)
    #todo complete this

def step(curr_state, pi, pTrue, rTrue):
    '''
    move forward by 1 time step in the MDP based on true params of MDP and subjective policy
    :param curr_state:
    :param pi:
    :param pTrue:
    :param rTrue:
    :return:
    '''
    action = pi[curr_state]
    reward = rTrue[curr_state, action]
    new_state = sampleDiscrete(pTrue[curr_state, action, :],1)

    return action, new_state, reward

def sampleFromDirichlet(alphas):
    assert np.min(alphas) > 0
    return np.random.dirichlet(alphas)





