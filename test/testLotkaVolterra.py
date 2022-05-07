import numpy as np
from matplotlib import pyplot as plt

# 区間の分割の設定
T = 20
n = 1000
h = T / n
t = np.arange(0, T, h)

a = 0.8
b = 0.5
c = 0.8
d = 1.0

# 方程式を定める関数、初期値の定義
f = lambda u, v, t=0: a * u - b * u * v
g = lambda u, v, t=0: c * u * v - d * v
u_0 = 2
v_0 = 1.1

# 結果を返すための配列の宣言
u = np.empty(n)
v = np.empty(n)
u[0] = u_0
v[0] = v_0

# 方程式を解くための反復計算
for i in range(n - 1):
    k_1 = h * f(u[i], v[i], t[i])
    k_2 = h * f(u[i] + k_1 / 2, v[i], t[i] + h / 2)
    k_3 = h * f(u[i] + k_2 / 2, v[i], t[i] + h / 2)
    k_4 = h * f(u[i] + k_3, v[i], t[i] + h)
    u[i + 1] = u[i] + 1 / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
    j_1 = h * g(u[i], v[i], t[i])
    j_2 = h * g(u[i], v[i] + j_1 / 2, t[i] + h / 2)
    j_3 = h * g(u[i], v[i] + j_2 / 2, t[i] + h / 2)
    j_4 = h * g(u[i], v[i] + j_3, t[i] + h)
    v[i + 1] = v[i] + 1 / 6 * (j_1 + 2 * j_2 + 2 * j_3 + j_4)

noise = 0.1
u_noisy = u + noise * np.std(u) * np.random.randn(u.shape[0])
v_noisy = v + noise * np.std(v) * np.random.randn(v.shape[0])

plt.plot(t, u_noisy, label='u_noisy')
plt.plot(t, v_noisy, label='v_noisy')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, concatenate
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
            loss = tf.reduce_mean(tf.reduce_mean(tf.square(y['u_data'] - y_pred[:, 0]))) + tf.reduce_mean(
                tf.square(y['v_data'] - y_pred[:, 1])) + tf.reduce_mean(
                tf.square(PPINs.preds(y_pred, x, self.a, self.b, self.c, self.d)))
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss)
        mae_metric.update_state(y['u_data'], y_pred[:, 0])
        mae_metric.update_state(y['v_data'], y_pred[:, 1])
        return {"loss": loss_tracker.result(), "mae": mae_metric.result(), 'a': self.a, "b": self.b, 'c': self.c,
                'd': self.d}

    @staticmethod
    def two_pred(y_pred, x, c, d):
        u = y_pred[:, 0]
        v = y_pred[:, 1]
        v_t = tf.gradients(v, x)
        return v_t - c * u * v + d * v

    @staticmethod
    def one_pred(y_pred, x, a, b):
        u = y_pred[:, 0]
        v = y_pred[:, 1]
        u_t = tf.gradients(u, x)
        return u_t - a * u + b * u * v

    @staticmethod
    def preds(y_pred, x, a, b, c, d):
        u = y_pred[:, 0]
        v = y_pred[:, 1]
        u_t = tf.gradients(u, x)
        v_t = tf.gradients(v, x)
        return v_t - c * u * v + d * v + u_t - a * u + b * u * v


import matplotlib.pyplot as plt


class PhysicsInformedNN:

    def __init__(self, t_array, u_data, v_data):
        self.__initial_NN()
        self.t_array = t_array
        self.u_data = u_data
        self.u_data_max = u_data.max()
        self.v_data = v_data
        self.v_data_max = v_data.max()

    def __initial_NN(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            t = Input(shape=(1,))
            z = Dense(1, activation="tanh")(t)
            for _ in range(48):
                z = Dense(20, activation='tanh')(z)
            z = Dense(2, activation="tanh")(z)

            self.model = PPINs(t, z)
            self.model.compile(optimizer="adam", metrics=['loss', 'mae', 'a', 'b'])

    def train(self, epochs=100):
        u_data = self.u_data / self.u_data_max
        v_data = self.v_data / self.v_data_max
        self.history = self.model.fit(self.t_array, y={"u_data": u_data, "v_data": v_data}, epochs=epochs)

    def plot_coefficient(self):
        plt.plot(self.history.history['a'], label='a')
        plt.plot(self.history.history['b'], label='b')
        plt.plot(self.history.history['c'], label='c')
        plt.plot(self.history.history['d'], label='d')
        plt.xlabel('train_num')
        plt.legend()

    def compare_numerical_ans(self, title):
        predict = self.model.predict(self.t_array)
        u_predict = self.u_data_max * predict[:, 0]
        v_predict = self.v_data_max * predict[:, 1]
        plt.plot(u, linestyle='--', label='numerical calculation', color='red')
        plt.plot(u_predict, linestyle='-.', label='neural network', color='blue')
        plt.plot(v, linestyle='--', color='red')
        plt.plot(v_predict, linestyle='-.', color='blue')
        plt.legend()

    def print_coeffisient(self):
        print('a={0}, b={1}, c={2}, d={3}'.format(self.model.a.numpy(), self.model.b.numpy(), self.model.c.numpy(),
                                                  self.model.d.numpy()))


if __name__ == '__main__':
    model = PhysicsInformedNN(t, u_noisy, v_noisy)
    model.train(5)
    model.compare_numerical_ans('lotka-volterra')
