from N0_prerequisites import *
from os import walk as oswalk
import pickle as pk
from numpy import array as arr

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
    result = pd.DataFrame(d[frame_in:-frame_out], columns=['inp','outp'])
    inp = result['inp'].tolist()
    oup = result['outp'].tolist()
    return inp, oup

def prep_time(frame_in, frame_out):
    sets = []
    for subdir, dirs, files in oswalk("./dataset/tser"):
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

def split(dset_inp, dset_oup, ratio_test):
    lenh = len(dset_inp)
    index_test = int(lenh * ratio_test)
    train_inp = arr(dset_inp[:index_test])
    train_oup = arr(dset_oup[:index_test])
    test_oup = arr(dset_oup[index_test:])
    test_inp = arr(dset_inp[index_test:])
    return train_inp, train_oup, test_inp, test_oup

def setup(test_ratio):
    frame_in = 24
    frame_out = 2
    sample_range = [500, 2000]
    dat_loc = "./dataset/tser/dat-{0}-{1}".format(frame_in,frame_out)
    sets = load_data(dat_loc, frame_in, frame_out, sample_range)
    sets_split = []
    for s in sets:
        sets_split.append(split(s[0],s[1],test_ratio))
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