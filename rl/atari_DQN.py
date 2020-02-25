import tensorflow as tf
import numpy as np
from typing import *
from . import agent
from utils.utils import *
from utils.tf_utils import *
import random

class AtariDQNAgent(agent.Agent):
    def __init__(self, env: gym.Env, hparams: Dict, sess :tf.Session, scope = "atariDQN"):
        self._sess = sess
        self.env = env
        self.epsilon = hparams['epsilon']
        self.input_frames = tf.placeholder(shape=[None, 84,84,4], dtype=tf.int32, name="input_frames")
        self.target_qvals = tf.placeholder(shape=[None], dtype=tf.float32, name="target_labels")
        self.actions_selected = tf.placeholder(shape=[None], dtype=tf.int32, name="actions_played")
        batch_size = tf.shape(self.input_frames)[0]
        input_grayscale = tf.cast(self.input_frames, tf.float32)/255.0

        conv1 = tf.contrib.layers.conv2d(
            input_grayscale, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        self.conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        flattened_conv3 = tf.contrib.layers.flatten(self.conv3)

        temp = fully_connected(flattened_conv3, 512)
        self.pred = fully_connected(temp, hparams['num_actions'], scope="fcn2")
        self.action = tf.argmax(self.pred, axis=1)
        indices = tf.range(0, tf.shape(self.actions_selected)[0]) * env.action_space.n + self.actions_selected
        prob_of_actions_selected = tf.gather(tf.reshape(self.pred,[-1]), indices)

        self.loss = tf.reduce_mean(tf.squared_difference(tf.to_float(self.actions_selected), prob_of_actions_selected))
        self.opt = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6) #hyperparams courtesy of dennybritz

        self.train_op = self.opt.minimize(self.loss)
        return

    def act(self, observation):
        if random.random() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        action = self._sess.run([self.action], feed_dict={self.input_frames : observation})
        return action[0]


    def train_step(self, obs, acts, advantages):
        feed_dict = {self.input_frames: obs,
                     self.actions_selected: acts,
                     self.target_qvals: advantages}
        loss, _ = self._sess.run([self.loss, self.train_op], feed_dict)
        return loss

