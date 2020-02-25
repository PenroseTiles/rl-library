import tensorflow as tf

from utils.utils import *


class PolicyGradientAgent(Agent):
    def __init__(self, env: gym.Env, hparams: Dict, sess: tf.Session):
        n = env.observation_space.shape[0]
        self._sess = sess

        # dtype = tf.float32 if type(env.observation_space) == gym.spaces.Box else tf.int32
        self._input = tf.placeholder(shape=[None, hparams['num_states']], dtype=tf.float32, name="input")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.advantages = tf.placeholder(shape=[None,1], dtype=tf.float32, name="rewards")

        W1 = tf.get_variable("W1",[n,hparams['hidden_size']],dtype=tf.float32)
        bias1 = tf.get_variable("b1", [hparams['hidden_size']], dtype=tf.float32)
        fc1 = tf.nn.relu(tf.matmul(self._input, W1, name="matmul1") + bias1)

        W2 = tf.get_variable("W2",[hparams['hidden_size'],hparams['num_actions']],dtype=tf.float32)
        bias2 = tf.get_variable("b2",[hparams['num_actions']],dtype=tf.float32)
        logits = tf.nn.relu(tf.matmul(fc1, W2, name="matmul2") + bias2)

        log_probs = tf.log(tf.nn.softmax(logits, name="actions_softmax"))

        indices = tf.range(0,tf.shape(log_probs)[0]) * hparams['num_actions']
        self.indices = indices + self.actions
        act_probs = tf.gather(tf.reshape(log_probs,[-1]),indices)


        #we're doing ax + by + cz instead of (a+b+c)*(x+y+z). correct this. use baselines and temporal info
        self.loss = -tf.reduce_sum(tf.multiply(act_probs, self.advantages))

        optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
        self._train = optimizer.minimize(self.loss)

        self._sample = tf.reshape(tf.multinomial(logits,1), [])


    def act(self, observation):
        return self._sess.run(self._sample, feed_dict={self._input: [observation]})

    def train_step(self, obs, advantages, acts):
        feed_dict = {self._input: obs,
                     self.actions: acts,
                     self.advantages: advantages}

        self._sess.run(self._train, feed_dict)


def gen_batch_of_rollouts(env, agent, eparams):
    batch_obs, batch_rews, batch_acts = [],[],[]
    for episode in range(eparams['ep_per_batch']):
        observations, rewards, actions =  policy_rollout(env, agent)
        batch_obs.extend(observations)
        batch_acts.extend(actions)
        batch_rews.extend(rewards)
    return batch_obs, batch_rews, batch_acts

def main():
    env = gym.make("CartPole-v0")
    hparams = {
        'num_states': env.observation_space.shape[0],
        'num_actions': env.action_space.n,
        'hidden_size': 32,
        'learning_rate': 0.1
    }

    eparams = {
            'num_batches': 40,
            'ep_per_batch': 1000
    }

    with tf.Graph().as_default(), tf.Session() as sess:
        agent = PolicyGradientAgent(env, hparams, sess)

        sess.run(tf.global_variables_initializer())
        for batch in range(eparams['num_batches']):
            obs, rews, acts = gen_batch_of_rollouts(env, agent, eparams)

            #baseline and avoid div-by-zero
            rews = np.reshape((np.array(rews) - np.mean(rews))/(np.std(rews) + 1e-10),[len(rews),1])
            agent.train_step(obs, rews, acts)

if __name__ == '__main__':
    main()