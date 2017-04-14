import objectives
import tensorflow as tf
import numpy as np
sess = tf.Session()

x = tf.random_normal([32, 6], mean=-1, stddev=4)
y = tf.random_normal([32, 6], mean=-1, stddev=4)
print('WTF')

# loss1 = objectives.huber_loss(x,y)
# loss2 = objectives.mean_huber_loss(x,y)

# print(sess.run(loss1))
# print(sess.run(loss2))

# a = tf.cond(tf.less(tf.abs(x-y), 0.5), lambda: tf.abs(x-y), lambda: tf.square(x-y))
loss = objectives.mean_huber_loss(x, y, max_grad=1.)
print(sess.run(loss))
