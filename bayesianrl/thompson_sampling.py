import bayesianrl.utils as utils
import numpy as np
import numpy.random as nr
import rl.value_iteration as va

def noise(shape):
    return nr.randn(*shape)


def updateValueAndPolicy(transition_freqs, alpha, reward, gamma):
    num_states, num_actions, num_states = transition_freqs.shape
    tp = np.zeros((num_states, num_actions, num_states))
    for state in range(num_states):
        for action in range(num_actions):
            tp[state, action] = nr.dirichlet(transition_freqs[state, action] + alpha, size=1)

    return va.valueIterationTillConvergence(tp, reward, gamma)

def actAndUpdate(transition_freqs, reward, reward_value, last_state, last_action, next_state, value_table, alphas, gamma, updated: np.ndarray, terminal=True, update=True):
    num_states, num_actions, num_states = transition_freqs.shape
    #updated has same shape

    if reward_value is None:
        action = nr.randint(num_actions)
        return action, value_table

    if not updated[last_state, last_action, next_state]:
        updated[last_state, last_action, next_state] = True
        reward[last_state, last_action, next_state] = reward_value

    if update:
        value_table = updateValueAndPolicy(transition_freqs, alphas, reward, gamma)

    transition_freqs[last_state, last_action, next_state] += 1
    action = np.argmax(value_table[next_state] + noise(num_actions))
    return action, value_table


