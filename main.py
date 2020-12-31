import pandas
import keras
from NN_Class import *
from data_preprocessing import *
from parameters import *
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample


if __name__ == '__main__':
    # combine and shuffle data
    all_records_NR_ARR_labeled_shuffled = combine_shufle_signal(all_records_NR_labeled_reduced,
                                                                all_records_ARR_labeled_reduced)


    all_record_NR_ARR_labeled_x = all_records_NR_ARR_labeled_shuffled[:, 0]
    all_record_NR_ARR_labeled_y = all_records_NR_ARR_labeled_shuffled[:, 1]

    # resample data using fourrier approach
    all_record_NR_ARR_labeled_x = resample(all_record_NR_ARR_labeled_x,
                                           int(all_record_NR_ARR_labeled_x.shape[0] / 100))


    # resample targets
    all_record_NR_ARR_labeled_y = np.zeros((7200,1))
    label = 0
    for i in range(0,7190,10):
        if (label == 0):
            all_record_NR_ARR_labeled_y[i:i+10] = 0
            label = 1
        else:
            all_record_NR_ARR_labeled_y[i:i + 10] = 1
            label = 0

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(all_record_NR_ARR_labeled_x,
                                                                all_record_NR_ARR_labeled_y,
                                                                VALIDATION_RATE,
                                                                TEST_RATE)

    #scale data
    scaler = MinMaxScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    train_gen = batch_generator(TIME_STEPS, x_train_scaled, y_train, BATCH_SIZE)
    val_gen = batch_generator(TIME_STEPS, x_val_scaled, y_val, BATCH_SIZE)

    nn_model = NN(INPUT_SHAPE, STEPS_PER_EPOCH, EPOCHS, VALIDATION_STEPS, train_gen, val_gen)
    nn_model.create_model()
    nn_model.compile_model()
    nn_model.fit_model()


