import tensorflow as tf
from tensorflow import keras

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")


class PPINs(keras.Model):

    def __init__(self, *args, **kwargs):
        super(PPINs, self).__init__(*args, **kwargs)
        self.a = tf.Variable([0], dtype=tf.float32)
        self.b = tf.Variable([0], dtype=tf.float32)
        self.c = tf.Variable([0], dtype=tf.float32)
        self.d = tf.Variable([0], dtype=tf.float32)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.reduce_mean(tf.square(y - y_pred)) + tf.reduce_mean(
                tf.square(PPINs.f_pred(y_pred, x, self.a, self.b, self.c, self.d))) + tf.reduce_mean(
                tf.square(y_pred[:, 0] - y_pred[:, -1]))
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result(), 'a': self.a, "b": self.b, 'c': self.c,
                'd': self.d}

    @staticmethod
    def f_pred(y_pred, x, a, b, c, d):
        u = y_pred
        u_x, u_t = tf.gradients(y_pred, x[0]), tf.gradients(y_pred, x[1])
        u_xx = tf.gradients(u_x, x[0])
        u_xxx = tf.gradients(u_xx, x[0])
        return u_t + a * u_x + b * u_xx + c * u_xxx + d * u * u_x


