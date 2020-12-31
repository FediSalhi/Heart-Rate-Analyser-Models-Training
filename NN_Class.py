from keras import Sequential
from keras.layers import Dense
from keras.activations import relu
from keras.activations import sigmoid
from keras.optimizers import Adam
from data_preprocessing import *
from keras.layers import BatchNormalization


class NN:
    def __init__(self, input_shape, steps_per_epoch, epochs, validation_steps,
                 train_gen, val_gen):
        self.input_shape = input_shape
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.validation_steps = validation_steps
        self.train_gen = train_gen
        self.valid_gen = val_gen
        self.model = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(128, 'tanh', input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Dense(64, 'linear'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, 'relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1, 'sigmoid'))

    def compile_model(self):

        self.model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

        self.model.summary()

    def fit_model(self):
        self.model.fit_generator(generator=self.train_gen,
                                 validation_data=self.valid_gen,
                                 epochs=self.epochs,
                                 steps_per_epoch=self.steps_per_epoch,
                                 validation_steps=self.validation_steps,
                                 shuffle=False)

    def show_summary(self):
        self.model.summary()

