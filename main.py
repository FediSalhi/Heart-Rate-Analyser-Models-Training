import pandas
import keras
from NN_Class import *
from data_preprocessing import *
from parameters import *


if __name__ == '__main__':
    all_record_NR_ARR_labeled_x = all_record_NR_ARR_labeled[:, 0]
    all_record_NR_ARR_labeled_y = all_record_NR_ARR_labeled[:, 1]

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(all_record_NR_ARR_labeled_x,
                                                                all_record_NR_ARR_labeled_y,
                                                                VALIDATION_RATE,
                                                                TEST_RATE)

    train_gen = batch_generator(TIME_STEPS, x_train, y_train, BATCH_SIZE)
    val_gen = batch_generator(TIME_STEPS, x_val, y_val, BATCH_SIZE)

    nn_model = NN(INPUT_SHAPE, STEPS_PER_EPOCH, EPOCHS, VALIDATION_STEPS, train_gen, val_gen)
    nn_model.create_model()
    nn_model.compile_model()
    nn_model.fit_model()


