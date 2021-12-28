from N0_prerequisites import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#There are two datasets for this experiment:
#1. Fashion-MNIST = Classification task
#2. NY Taxi data for July 2019 = Time series foreasting task

# Initializations
img_rows, img_cols = 28, 28

def fetch():
    dset = pd.read_csv("./dataset/archive/fmni.csv")
    dset.sample(frac=1)
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