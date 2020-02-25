
class Agent():
    def act(self, observation):
        raise NotImplementedError

    def train_step(self, obs, acts, advantages):
        raise NotImplementedError