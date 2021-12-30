from N0_prerequisites import *
from os import walk as oswalk
import pickle as pk
from numpy import array as arr
from sklearn.metrics import mean_squared_error

# TIME SERIES
# Hourly data from 2015.01.01 00.00, source = https://www.kaggle.com/francoisraucent/western-europe-power-consumption
def unnegate(num):
    if num < 0:
        num = 0
    return num

def make_set(dset, frame_in, frame_out):
    global m
    d = []
    idx = -frame_in + 1
    for i in range(len(dset)):
        dj = []
        dk = []
        for j in range(frame_in):
            dj.append(dset.iloc[unnegate(idx + j), 1])
        for k in range(frame_out):
            try:
                dk.append(dset.iloc[unnegate(idx + frame_in + k), 1])
            except:
                break
        d.append([dj,dk])
        idx += 1
    result = pd.DataFrame(d[frame_in:-frame_out], columns=['inp','oup'])
    inp = result['inp'].tolist()
    oup = result['oup'].tolist()
    return inp, oup

def prep_time(frame_in, frame_out):
    sets = []
    for subdir, dirs, files in oswalk("./dataset/tser/norm"):
        for filename in files:
            filepath = subdir + '/' + filename
            dat = pd.read_csv(filepath)
            dat = make_set(dat, frame_in, frame_out)
            sets.append(dat)
    return sets

def load_data(dat_loc, frame_in, frame_out, sample_range):
    try:
        sets = pk.load(open(dat_loc, "rb"))
    except FileNotFoundError:
        sets = prep_time(frame_in, frame_out)
        pk.dump(sets, open(dat_loc, "wb"))
    return sets

def split(dset_inp, dset_oup, ratio_test, frame_in, frame_out):
    lenh = len(dset_inp)
    index_test = int(lenh * ratio_test)
    test_inp = np.reshape(arr(dset_inp[:index_test]),(index_test,1,frame_in,1))
    test_oup = arr(dset_oup[:index_test])
    train_oup = arr(dset_oup[index_test:])
    train_inp = np.reshape(arr(dset_inp[index_test:]),(lenh - index_test,1,frame_in,1))
    print(train_inp.shape, train_oup.shape, test_oup.shape, test_inp.shape)
    return train_inp, train_oup, test_inp, test_oup

def setup(test_ratio):
    frame_in = 24
    frame_out = 2
    sample_range = [500, 2000]
    dat_loc = "./dataset/tser/norm/dat-{0}-{1}".format(frame_in,frame_out)
    sets = load_data(dat_loc, frame_in, frame_out, sample_range)
    sets_split = []
    for s in sets:
        sets_split.append(split(s[0],s[1],test_ratio, frame_in, frame_out))
    return sets_split

def compound(sets_split):
    train_inp = []
    train_oup = []
    test_inp = []
    test_oup = []
    for s in sets_split:
        train_inp.extend(s[0])
        train_oup.extend(s[1])
        test_inp.extend(s[2])
        test_oup.extend(s[3])
    return arr(train_inp), arr(train_oup), arr(test_inp), arr(test_oup)


class Prediction:
    def __init__(self, inp, outp, model, frame_in, frame_out):
        self.inp = inp
        self.model = model
        self.outp = outp
        self.pred = []
        self.frame_in = frame_in
        self.rmse = None
        self.frame_out = frame_out
        self.in_percent = 0

    def predict(self):
        for to_test_idx in range(len(self.inp)):
            x_input = self.inp[to_test_idx]
            x_input = x_input.reshape((1, 1, self.frame_in, 1))
            yhat = self.model.model.predict(x_input, verbose=0).tolist()[0]
            for i in range(len(yhat)):
                yhat[i] = max(yhat[i],0)
            self.pred.append(yhat)
        self.rmse = (mean_squared_error(self.pred, self.outp) / self.frame_out) ** 0.5  

    def summary(self, label="", verbose=True):
        sum_dat = 0
        len_dat = 0
        for i in self.inp:
            for j in i:
                sum_dat += j[0]
                len_dat += 1
        average_bin_value = sum_dat/len_dat
        self.in_percent = self.rmse/average_bin_value*10000//1/100

        if verbose:
            #print("RMSE " + label + " = " + str(int(self.rmse * 100)/100), end = ' | ')
            print("Total demand " + label + " = " + str(sum_dat), end =' || ')
            print("Average demand " + label + " = " + str(int(average_bin_value * 100)/100), end=' || ')
            print("RMSE " + label + " = " + str(self.in_percent) + "%", end=' || ')

    def extract(self):
        return self.pred