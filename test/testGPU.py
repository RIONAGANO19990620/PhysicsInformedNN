import unittest
from tensorflow import keras


class testGPU(unittest.TestCase):

    def setUp(self) -> None:

        self.model = keras.Sequential()
        self.model.add(keras.layers.Flatten(input_shape=(28, 28)))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        for _ in range(3):
            self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def test_fit(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        self.model.fit(train_images, train_labels, epochs=1)
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)


if __name__ == '__main__':
    unittest.main()
