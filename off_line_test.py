import pandas
import keras
from NN_Class import *
from data_preprocessing import *
from parameters import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.models import load_model

all_records_NR_ARR_labeled_shuffled = combine_shufle_signal(all_records_NR_labeled_reduced,
                                                                all_records_ARR_labeled_reduced)


all_record_NR_ARR_labeled_x = all_records_NR_ARR_labeled_shuffled[:, 0]
all_record_NR_ARR_labeled_y = all_records_NR_ARR_labeled_shuffled[:, 1]

#reduce data
all_record_NR_ARR_labeled_x = all_record_NR_ARR_labeled_x[0:NUMBER_OF_REDUCED_DATA]
all_record_NR_ARR_labeled_y = all_record_NR_ARR_labeled_y[0:NUMBER_OF_REDUCED_DATA]



x_train, y_train, x_val, y_val, x_test, y_test = split_data(all_record_NR_ARR_labeled_x,
                                                                all_record_NR_ARR_labeled_y,
                                                                VALIDATION_RATE,
                                                                TEST_RATE)
#scale data
scaler = StandardScaler()
scaler = scaler.fit(x_train)


# Test script
# load model
model = load_model('C:\\Users\\Designer\\Desktop\\BLM5135_Final_Project\\Heart Rate Analyser Models Training\\model.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x_test = scaler.transform(x_test)

preds = []
for i in range(NUMBER_OF_TEST_CHUNKS):
    prediction = model.predict(x_test[int(i * TIME_STEPS): int(i * TIME_STEPS) + TIME_STEPS].reshape(1, TIME_STEPS, 1))
    preds.append(prediction)
preds = np.array(preds).reshape(6, -1)

#refactor y_test
targets = []
for i in range(1000, NUMBER_OF_TEST_CHUNKS*1000, 1000):
    targets.append(y_test[i])
    print(y_test[i])
targets.append(0)

print(targets)
print(preds)
x_axis = [x for x in range(6)]
plt.scatter(x_axis, preds, label='Predictions')
plt.scatter(x_axis, targets, label='Targets')
plt.title("Test Predictions: 1-Arrhythmia 0-Normal Rhythm")
plt.legend()
plt.xlabel("ECG Chunk Index")
plt.ylabel("Predictions")
plt.show()