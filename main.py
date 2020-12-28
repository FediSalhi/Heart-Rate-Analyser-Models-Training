import pandas
import keras
from NN_Class import *


if __name__ == '__main__':
    NN_model = NN((30, 1))
    NN_model.create_model()
    NN_model.compile_model()
    print("a")
