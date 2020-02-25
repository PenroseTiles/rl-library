from rl.agent import Agent
from utils.utils import *


class TabularQAgent(Agent):
    def __init__(self, env):
        self.table = np.zeros((env.observation_space.n, env.action_space.n))
        self.env = env
        return

    def __str__(self):
        print(self.table)
        return ""

    def act(self, observation):
        return np.argmax(self.table[observation,:])


    def train_step(self, obs, acts, advantages):
        raise NotImplementedError

def train(Qagent: TabularQAgent, env: gym.Env, hparams: Dict):
    s = env.reset()
    episodes, rAll = [], []
    for i in range(hparams['num_episodes']):
        for j in range(hparams['max_steps_per_episode']):
            env.render()
            #sample action a
            a = Qagent.act(s)

            if np.random.uniform(0,1) > hparams['epsilon']:
                a = env.action_space.sample()

            #sample next state from the environment
            s1, reward, done, info = env.step(a)

            episodes.append(s1)
            rAll.append(reward)
            if done:
                target = reward
                print(reward)
                s1 = env.reset()
            else:
                target = reward + hparams['gamma'] * np.max(Qagent.table[s1,:])
            Qagent.table[s,a] = hparams['alpha'] * target + (1 - hparams['alpha'])* Qagent.table[s,a]
            s = s1
        s1 = env.reset()
        s = s1
    return Qagent



if __name__ == "__main__":
    env_ = gym.make('FrozenLake-v0')
    hparams_ = {
        'epsilon' : 0.4,
        'gamma' : 0.95,
        'alpha' : 0.4,
        'num_episodes' : 10000,
        'max_steps_per_episode' : 99
    }
    Qagent_ = TabularQAgent(env_)
    Qagent_ = train(Qagent_, env_, hparams_)
    deterministic_test(Qagent_, env_)