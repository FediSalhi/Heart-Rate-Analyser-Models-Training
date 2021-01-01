from keras import Sequential
from keras.layers import Dense
from keras.activations import relu
from keras.activations import sigmoid
from keras.optimizers import Adam
from data_preprocessing import *
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten


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
        self.model.add(Conv1D(filters=4, kernel_size=5, padding='same', activation='relu', input_shape=(TIME_STEPS, 1)))
        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(Dropout(0.3))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(8, 'relu'))
        self.model.add(Dense(1, 'sigmoid'))

    def compile_model(self):

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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

