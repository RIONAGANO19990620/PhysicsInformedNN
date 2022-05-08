import numpy as np
import tensorflow as tf
from tensorflow import keras

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")


class PPINs(keras.Model):

    def __init__(self, *args, **kwargs):
        super(PPINs, self).__init__(*args, **kwargs)
        self.a = tf.Variable(np.random.randn(1) * 0.01, dtype=tf.float32)
        self.b = tf.Variable(np.random.randn(1) * 0.01, dtype=tf.float32)
        self.c = tf.Variable(np.random.randn(1) * 0.01, dtype=tf.float32)
        self.d = tf.Variable(np.random.randn(1) * 0.01, dtype=tf.float32)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            """loss_1 = tf.reduce_mean(tf.square(y - y_pred)) + tf.reduce_mean(tf.square(y_pred[0, :] - y[0, :])) \
                   + tf.reduce_mean(tf.square(y_pred[:, 0] - y[:, 0])) + tf.reduce_mean(
                tf.square(y_pred[:, -1] - y[:, -1]))
            loss_2 = tf.reduce_mean(
                tf.square(PPINs.f_pred(y_pred, x, self.a, self.b, self.c, self.d))) + loss_1
            loss = tf.cond(loss_1 < 0.01, lambda: loss_2, lambda: loss_1)"""
            loss = tf.reduce_mean(tf.square(y - y_pred)) + tf.reduce_mean(tf.square(y_pred[0, :] - y[0, :])) \
                   + tf.reduce_mean(tf.square(y_pred[:, 0] - y[:, 0])) + tf.reduce_mean(
                tf.square(y_pred[:, -1] - y[:, -1])) \
                   + tf.reduce_mean(tf.square(self.f_pred(y_pred, x, self.a, self.b, self.c, self.d)))
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result(), 'a': self.a, 'b': self.b, 'c': self.c,
                'd': self.d}

    @staticmethod
    def f_pred(u, x, a: tf.Variable, b: tf.Variable, c: tf.Variable, d: tf.Variable):
        x = x[0]
        t = x[1]
        u_x = tf.gradients(u, x)
        u_t = tf.gradients(u, t)
        u_xx = tf.gradients(u_x, x)
        u_xxx = tf.gradients(u_xx, x)
        return u_t + a * u_x + b * u_xx + c * u_xxx + d * u * u_x

    @staticmethod
    def kdv_format(y_pred, x, c, d):
        u = y_pred
        u_x, u_t = tf.gradients(y_pred, x[0]), tf.gradients(y_pred, x[1])
        u_xx = tf.gradients(u_x, x[0])
        u_xxx = tf.gradients(u_xx, x[0])
        return u_t + c * u_xxx + d * u * u_x
