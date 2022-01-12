import pandas as pd
import numpy as np
from random import randint
from matplotlib import pyplot as pl
from tensorflow.python.keras.callbacks import History
from time import time
from copy import deepcopy
from random import random as rdm

NaN = np.nan

class Node:
    def __init__(self):
        self.id_ = str.format("{:05d}",randint(0,10000))
        self.inp = None
        self.oup = None
        self.model = None

class Model:
    def __init__(self):
        self.model = None
        self.inp = None
        self.oup = None
        self.sum_dat = 0

    def setData(self,inp,oup):
        self.inp = inp
        self.oup = oup
        sum_dat = 0
        for i in self.inp:
            for j in i:
                sum_dat += j[0]
        self.sum_dat = sum_dat
    
    def summarize(self):
        self.model.summary()

    def train(self, label, session_num, verbose_=0, epochs_=1, validation_set_=None):
        start = time()
        if validation_set_ != None:
            self.model.fit(self.inp, self.oup, epochs=epochs_, verbose=verbose_, shuffle=True, validation_data=validation_set_)    
        else:
            self.model.fit(self.inp, self.oup, epochs=epochs_, verbose=verbose_, shuffle=True)
        end = time()
        s = 'Session ' + session_num + ' | Label: ' + label + ' | Epochs: ' + str(epochs_) + ' | Training volume: ' + str(len(self.inp)) + ' | Quantity: ' + str(self.sum_dat) + ' | Duration: ' + str(end-start)
        f = open("./temp/session-training.txt", 'a+')
        f.write(s + '\n')
        f.close()

    def trainW(self, label, session_num, verbose_=0, epochs_=1, validation_set_=None):
        start = time()
        if validation_set_ != None:
            qual = self.model.fit(self.inp, self.oup, epochs=epochs_, verbose=verbose_, shuffle=True, validation_data=validation_set_)    
        else:
            qual = self.model.fit(self.inp, self.oup, epochs=epochs_, verbose=verbose_, shuffle=True)
        end = time()
        s = 'Session ' + session_num + ' | Label: ' + label + ' | Epochs: ' + str(epochs_) + ' | Training volume: ' + str(len(self.inp)) + ' | Quantity: ' + str(self.sum_dat) + ' | Duration: ' + str(end-start)
        f = open("./temp/session-training.txt", 'a+')
        f.write(s + '\n')
        f.close()
        return qual

    def drawHist(self, factor):
        hist = self.history.history[factor]
        _, hist_ax = pl.subplots()
        hist_ax.set_yscale("log")
        hist_ax.plot(hist)

    def replaceModel(self, model):
        self.model = model
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

def getSession():
    f = open("./temp/session_num.txt", 'r')
    ses = int(f.read())
    f.close()
    f = open("./temp/session_num.txt", 'w')
    f.write(str(ses + 1))
    f.close()
    return str(ses + 1)