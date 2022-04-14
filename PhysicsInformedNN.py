import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, Dense, Input

from Model import PPINs


class PhysicsInformedNN:

    def __init__(self, x_array, t_array, teacher_data):
        self.__initial_NN()
        self.x_array = x_array
        self.t_array = t_array
        self.teacher_data = teacher_data
        self.teacher_data_max = self.teacher_data.max()

    def __initial_NN(self):
        input1 = Input(shape=(1,))
        input2 = Input(shape=(1,))
        x = Dense(1, activation="tanh")(input1)
        x = Model(inputs=input1, outputs=x)
        y = Dense(1, activation="tanh")(input2)
        y = Model(inputs=input2, outputs=y)
        combined = concatenate([x.output, y.output])
        z = Dense(20, activation="tanh")(combined)
        for _ in range(7):
            z = Dense(20, activation='tanh')(z)
        z = Dense(1, activation="tanh")(z)

        self.model = PPINs([x.input, y.input], z)
        self.model.compile(optimizer="adam", metrics=['loss', 'mae'])

    def train(self, epochs=100):
        input_data = self.__get_input_data()
        teacher_data = self.teacher_data / self.teacher_data_max
        self.history = self.model.fit(input_data, teacher_data, epochs=epochs)

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

    def save_plot_u(self, data, title):
        u_pred = self.teacher_data_max * self.model.predict(self.__get_input_data())
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
        plt.savefig('{}.png'.format(title))
        plt.close()
        plt.clf()

    def save_print_coeffisient(self, title):
        plt.title(title)
        plt.plot(self.history.history['a'], label='a')
        plt.plot(self.history.history['b'], label='b')
        plt.plot(self.history.history['c'], label='c')
        plt.plot(self.history.history['d'], label='d')
        plt.xlabel('train_num')
        plt.legend()
        plt.savefig('{}_coeffisient.png'.format(title))
        plt.close()
        plt.clf()

    def print_coeffisient(self):
        print('a={0}, b={1}, c={2}, d={3}'.format(self.model.a.numpy(), self.model.b.numpy(), self.model.c.numpy(),
                                                  self.model.d.numpy()))
