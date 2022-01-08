from N0_prerequisites import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

#There are two datasets for this experiment:
#1. Fashion-MNIST = Classification task
#2. NY Taxi data for July 2019 = Time series foreasting task

# Initializations
img_rows, img_cols = 28, 28

def fetch():
    dset = pd.read_csv("./dataset/archive/fmni.csv")
    dset.sample(frac=1).reset_index(drop=True)
    return dset

def normalize_portions(portions):
    portions_sum = sum(portions)
    portions_normal = []
    for i in portions:
        portions_normal.append(i/portions_sum)
    return portions_normal

def split(dset, test_ratio):
    inp = np.array(dset.iloc[:, 1:])
    oup = to_categorical(np.array(dset.iloc[:, 0]))

    train_inp, test_inp, train_oup, test_oup = train_test_split(inp, oup, test_size=test_ratio, random_state=13)

    train_inp = train_inp.reshape(train_inp.shape[0], img_rows, img_cols, 1)
    test_inp = test_inp.reshape(test_inp.shape[0], img_rows, img_cols, 1)

    train_inp = train_inp.astype('float32') / 255
    test_inp = test_inp.astype('float32') / 255

    return train_inp, test_inp, train_oup, test_oup

def distribute(dset, portions):
    sets = []
    cumulative = 0
    size = len(dset)
    for p in portions:
        d = dset[int(cumulative*size):int((cumulative+p)*size)]
        cumulative += p
        sets.append(d)
    return sets

# SEPARATE THIS FOR ML
#portions = portion of each node ([1,2,4]): node 1: 1/6, node 2: 2/6, node 3: 4/6
#test_in, test_out, train_in, train_out = prep_mnist()

class CNN(Model):
    def __init__(self):
        super().__init__()

    def setup(self):
        self.model = Sequential()
        self.model.add(Conv2D(6, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(12, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(12, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(12, activation='relu'))
        self.model.add(Dense(10))
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Multi-client construction
def buildPortions(count, dataset):
    portions = [int(2 ** (x + 1)) for x in range(count)]
    portions_idx = [int(e / sum(portions) * len(dataset)) for e in portions][::-1]
    portions = []
    index = 0
    for e in portions_idx:
        portions.append(dataset[index : (index + e)])
        index += e
    print(portions_idx)
    return portions

# Multi-client validation
def testMulti(test_inps_, test_oups_, model_):
    lenh = len(test_inps_)
    for i in range(lenh):
        if i in [0, lenh // 2, lenh - 1]:
            print(i, len(test_inps_[i]), len(test_oups_[i]))
            model_.model.evaluate(test_inps_[i], test_oups_[i])