import gym
import numpy as np
import numpy.random as nr
from typing import List, Tuple, Dict
from rl.agent import Agent
import time

def get_env_dynamics_network(env: gym.Env):
    raise NotImplementedError



def policy_rollout(env: gym.Env, agent):
    #run an episode
    observation, reward, done = env.reset(), 0, False
    obs, rews, acts = [],[],[]
    while not done:
       env.render()
       obs.append(observation)
       act = agent.act(observation)
       observation, reward, done, _ = env.step(act)
       rews.append(reward)
       acts.append(act)
    return obs, rews, acts

def process_current_discounted_return(rewards: List, gamma: float) -> float:
    temp = 0.
    for reward in reversed(rewards):
        temp = gamma * temp + reward
    return temp

def process_all_discounted_returns(rewards: List, gamma: float) -> List:
    disc_rewards = [0 in range(len(rewards))]
    temp = 0.
    for k, reward in enumerate(reversed(rewards)):
        temp = gamma* temp + reward
        disc_rewards[-(k+1)] = temp
    return disc_rewards

def process_rewards_cartpole(rewards: List):
    #reward is the time survived
    return [len(rewards)] * len(rewards)

def deterministic_test(agent: Agent, env):
    s = env.reset()
    actions = []
    while True:
        env.render()
        a = agent.act(s)
        s1, reward, done, _= env.step(a)
        actions.append(a)
        time.sleep(0.5)
        s = s1
        if done:
            time.sleep(2)
            print(agent)
            print(s)
            print(reward)
            exit(0)