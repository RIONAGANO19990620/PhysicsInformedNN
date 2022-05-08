from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, Dense, Input
import tensorflow as tf

import matplotlib.animation as animation

from Execution.Model import PPINs


class PhysicsInformedNN:

    def __init__(self, x_array, t_array, teacher_data):
        self.x_array = (x_array - x_array.min()) / (x_array.max() - x_array.min())
        self.t_array = (t_array - t_array.min()) / (t_array.max() - t_array.min())
        self.teacher_data = teacher_data
        self.__initial_NN()

    def __initial_NN(self):
        input1 = Input(shape=(1,))
        input2 = Input(shape=(1,))
        x = Dense(1, activation="tanh")(input1)
        x = Model(inputs=input1, outputs=x)
        y = Dense(1, activation="tanh")(input2)
        y = Model(inputs=input2, outputs=y)
        combined = concatenate([x.output, y.output])
        z = Dense(20, activation="tanh")(combined)
        z = Dense(20, activation='tanh')(z)
        z = Dense(20, activation='tanh')(z)
        z = Dense(20, activation='tanh')(z)
        z = Dense(20, activation='tanh')(z)
        z = Dense(20, activation='tanh')(z)
        z = Dense(20, activation='tanh')(z)
        z = Dense(20, activation='tanh')(z)
        z = Dense(20, activation='tanh')(z)
        z = Dense(1, activation="tanh")(z)

        self.model = PPINs([x.input, y.input], z)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001),
            metrics=['loss', 'mae', 'a', 'b', 'c', 'd'])

    def train(self, epochs=100):
        input_data = self.__get_input_data()
        self.history = self.model.fit(input_data, self.teacher_data, epochs=epochs)

    def plot_coefficient(self):
        plt.plot(self.history.history['a'], label='a')
        plt.plot(self.history.history['b'], label='b')
        plt.plot(self.history.history['c'], label='c')
        plt.plot(self.history.history['d'], label='d')
        plt.xlabel('train_num')
        plt.legend()

    def __get_input_data(self):
        t = self.t_array.flatten()[:, None]
        x = self.x_array.flatten()[:, None]
        X, T = np.meshgrid(x, t)
        return [X.flatten()[:, None], T.flatten()[:, None]]

    def save_plot_u(self, data, title, path: Path):
        u_pred = self.model.predict(self.__get_input_data())
        u_pred_reshaped = u_pred.reshape(len(self.t_array), len(self.x_array))
        for t_n in range(len(self.t_array)):
            plt.title(title)
            plt.xlabel('x')
            plt.ylabel('u')
            if t_n % (len(self.t_array) // 5) == 0 or t_n == len(self.t_array) - 1:
                plt.plot(self.x_array, data[t_n], linestyle='--', color='red')
                plt.plot(self.x_array, u_pred_reshaped[t_n], linestyle='-.', color='blue')
        plt.plot(self.x_array, data[t_n], label='numerical calculation', linestyle='--', color='red')
        plt.plot(self.x_array, u_pred_reshaped[t_n], label='neural network', linestyle='-.', color='blue')
        plt.savefig(path / '{}.png'.format(title))
        plt.close()
        plt.clf()

    def save_plot_coeffisient(self, title, path: Path):
        plt.title(title)
        plt.plot(self.history.history['a'], label='a')
        plt.plot(self.history.history['b'], label='b')
        plt.plot(self.history.history['c'], label='c')
        plt.plot(self.history.history['d'], label='d')
        plt.xlabel('train_num')
        plt.legend()
        plt.savefig(path / '{}_coeffisient.png'.format(title))
        plt.close()
        plt.clf()

    def print_coeffisient(self, path):
        text = 'a={0}, b={1}, c={2}, d={3}'.format(self.model.a.numpy(), self.model.b.numpy(), self.model.c.numpy(),
                                                   self.model.d.numpy())
        file = open(path / 'data.txt', 'w')
        file.write(text)
        file.close()

    def save_plot_gif(self, path: Path):
        u_pred = self.model.predict(self.__get_input_data())
        u_pred_reshaped = u_pred.reshape(len(self.t_array), len(self.x_array))
        teacher_data_reshaped = self.teacher_data.reshape(len(self.t_array), len(self.x_array))
        fig, ax = plt.subplots()
        ims = []
        for t_n in range(len(self.t_array)):
            pred_im = ax.plot(u_pred_reshaped[t_n], color="red")
            teacher_im = ax.plot(teacher_data_reshaped[t_n], color="blue")
            ims.append(pred_im + teacher_im)
        ax.set_xlabel('x')
        ani = animation.ArtistAnimation(fig, ims, interval=1)
        ani.save(str(path / 'pred_data.gif'))
        plt.close(fig)
        plt.clf()
