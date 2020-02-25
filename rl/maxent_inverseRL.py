import numpy as np
from rl.value_iteration import *
from itertools import product
def getFeatureExpectationOverAllTrajectories(state_features, trajectories):
    '''

    :param state_features: Nxdim
    :param trajectories: TxLx2
    :return: dim
    '''

    only_states = np.reshape(trajectories[:,:,0],-1)
    return np.mean(state_features[only_states], axis=0)

def getExpectedStateVisitationFreqs(num_states, R, num_actions, gamma, transition_probs, trajectories):
    num_trajs, traj_length,_ = trajectories.shape
    pi = getStochasticPolicy(transition_probs,R,None, gamma)
    start_state_count = np.zeros(shape=(num_states,))

    all_start_states = trajectories[:,0,0].reshape([-1]).tolist()
    for state in all_start_states:
        start_state_count[state] += 1
    start_state_p = start_state_count/trajectories.shape[0]
    exp_svf = np.tile(start_state_p, (trajectories.shape[1],1)).T

    for traj in range(1, trajectories.shape[1]):
        exp_svf[:,traj] = 0
        for i,j,k in product(range(num_states), range(num_actions), range(num_states)):
            exp_svf[k, traj] += (exp_svf[i, traj-1] * pi[i,j] * transition_probs[i,j,k])
    return exp_svf.sum(axis=1)


def getSVF(num_states: int, trajectories: np.ndarray):
    svf = np.zeros(num_states)
    only_states = trajectories[:,:,0].reshape([-1]).tolist()
    for state in only_states:
        svf[state] +=1
    svf = svf/trajectories.shape[0]
    return svf

def inverseReinforcementLearn(state_features, gamma, transition_probs, trajectories, num_epochs, lr):
    num_states, num_actions, _ = transition_probs.shape
    _, state_dim = state_features.shape
    expected_features = getFeatureExpectationOverAllTrajectories(state_features, trajectories)
    theta = np.random.uniform(size=(state_dim))

    #maximize the log probability of visited states
    for i in range(num_epochs):
        r = np.dot(state_features, theta)
        expected_SVF = getExpectedStateVisitationFreqs(
            num_states, r, num_actions, gamma,transition_probs, trajectories)
        gradients = expected_features - state_features.dot(expected_SVF)
        theta = theta + lr*gradients

    return state_features.dot(theta).reshape([num_states,])

