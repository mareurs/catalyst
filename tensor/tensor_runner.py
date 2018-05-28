import tensorflow as tf
import os

net = tf.InteractiveSession()
saver = tf.train.Saver()
save_dir = 'checkpoints/'
save_path = os.path.join(save_dir, 'best_validation')


saver.restore(sess=net, save_path=save_path)
pred = net.run(out, feed_dict={X: X_test})
