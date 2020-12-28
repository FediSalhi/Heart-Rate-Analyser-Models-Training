from keras import Sequential
from keras.layers import Dense
from keras.activations import relu
from keras.activations import sigmoid
from keras.optimizers import Adam


class NN:
    def __init__(self, input_shape):
        self.input_shape = input_shape


        self.model = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, 'relu', input_shape=self.input_shape))
        self.model.add(Dense(64, 'sigmoid'))
        self.model.add(Dense(1, 'sigmoid'))

    def compile_model(self):
        self.model.summary()