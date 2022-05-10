import numpy as np
import tensorflow as tf
from tensorflow import keras

from Execution import NormalizedData

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.RootMeanSquaredError(name="mae")


class PPINs(keras.Model):

    def __init__(self, normalized_data: NormalizedData, *args, **kwargs):
        super(PPINs, self).__init__(*args, **kwargs)
        self.a = tf.Variable([0], dtype=tf.float32)
        self.b = tf.Variable([0], dtype=tf.float32)
        self.c = tf.Variable([0], dtype=tf.float32)
        self.d = tf.Variable([0], dtype=tf.float32)
        self.normalized_data = normalized_data

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            """loss_1 = tf.reduce_mean(tf.square(y - y_pred)) + tf.reduce_mean(
                tf.square(y_pred[:, 0] - y_pred[:, -1]))
            loss_2 = tf.reduce_mean(
                tf.square(PPINs.f_pred(y_pred, x, self.a, self.b, self.c, self.d)))
            loss = tf.cond(loss_1 < 0.0155, lambda: loss_2, lambda: loss_1)"""
            loss = tf.reduce_mean(tf.square(y - y_pred)) + tf.reduce_mean(tf.square(
                self.all_f(y_pred, x, self.a, self.b, self.c, self.d, self.normalized_data.u_max,
                         self.normalized_data.u_min, self.normalized_data.t_max,
                         self.normalized_data.t_min, self.normalized_data.x_max,
                         self.normalized_data.x_min))) + tf.reduce_mean(
                tf.square(y_pred[0, :] - y[0, :])) \
                   + tf.reduce_mean(tf.square(y_pred[:, 0] - y[:, 0])) + tf.reduce_mean(
                tf.square(y_pred[:, -1] - y[:, -1]))
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result(), 'a': self.a, 'b': self.b, 'c': self.c,
                'd': self.d}

    @staticmethod
    def all_f(y_pred, x, a, b, c, d, u_max, u_min, t_max, t_min, x_max, x_min):
        u = y_pred
        u_x, u_t = tf.gradients(y_pred, x[0]), tf.gradients(y_pred, x[1])
        u_xx = tf.gradients(u_x, x[0])
        u_xxx = tf.gradients(u_xx, x[0])
        return tf.divide(u_t, t_max - t_min) + a * tf.divide(u_x, x_max - x_min) + b * tf.divide(u_xx, (
                x_max - x_min) ** 2) + c * tf.divide(u_xxx, (x_max - x_min) ** 3) + d * tf.divide(
            u * (u_max - u_min) + u_min,
            x_max - x_min) * u_x

    @staticmethod
    def kdv(y_pred, x, a, b, c, d, u_max, u_min, t_max, t_min, x_max, x_min):
        u = y_pred
        u_x, u_t = tf.gradients(y_pred, x[0]), tf.gradients(y_pred, x[1])
        u_xx = tf.gradients(u_x, x[0])
        u_xxx = tf.gradients(u_xx, x[0])
        return tf.divide(u_t, t_max - t_min) + c * tf.divide(u_xxx, (x_max - x_min) ** 3) + d * tf.divide(
            u * (u_max - u_min) + u_min,
            x_max - x_min) * u_x

    @staticmethod
    def advection_diffusion(y_pred, x, a, b, c, d, u_max, u_min, t_max, t_min, x_max, x_min):
        u = y_pred
        u_x, u_t = tf.gradients(y_pred, x[0]), tf.gradients(y_pred, x[1])
        u_xx = tf.gradients(u_x, x[0])
        return tf.divide(u_t, t_max - t_min) + a * tf.divide(u_x, x_max - x_min) + b * tf.divide(u_xx,
                                                                                                 (x_max - x_min) ** 2)

    @staticmethod
    def burgers(y_pred, x, a, b, c, d, u_max, u_min, t_max, t_min, x_max, x_min):
        u = y_pred
        u_x, u_t = tf.gradients(y_pred, x[0]), tf.gradients(y_pred, x[1])
        u_xx = tf.gradients(u_x, x[0])
        return tf.divide(u_t, t_max - t_min) + b * tf.divide(u_xx, (x_max - x_min) ** 2) + d * u_x * tf.divide(
            u * (u_max - u_min) + u_min, x_max - x_min)
