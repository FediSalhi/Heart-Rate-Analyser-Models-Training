from keras.models import load_model
import numpy as np
import sys

# x = [i for i in range(1000)]
# x = np.array(x)
# x = x.reshape(1,1000,1)


model = load_model('C:\\Users\\Designer\\Desktop\\BLM5135_Final_Project\\Heart Rate Analyser Models Training\\model.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

input = (sys.argv[1]).split(' ')

input = input[:-1]
for element_index in range(len(input)):
    if (input[element_index] != ''):
        input[element_index] = input[element_index].replace(',', '.')
        input[element_index] = float(input[element_index])

input_np = (np.array(input)).reshape(1, 1000, 1)

prediction = model.predict(input_np)

output = "not_changed"
if (prediction[0] > 0.5):
    output = "Arrhythmia"
else:
    output = "Normal Rhythm"

print(output)
