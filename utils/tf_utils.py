import tensorflow as tf
import numpy as np

def compute_entropy(p):
    return -tf.reduce_sum(p * tf.log(p))

def fully_connected(x, num_output,scope="fcn", reuse = False):
    with tf.variable_scope(scope,reuse=reuse):
        W = tf.get_variable(name='W', shape=[x._shape.as_list()[1], num_output])
        b = tf.get_variable(name='b', shape=[num_output])
        return tf.nn.xw_plus_b(x,W,b, name="linear")