import pandas as pd
import numpy as np
from random import randint
from matplotlib import pyplot as pl
from tensorflow.python.keras.callbacks import History

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
        self.history = History()

    def setData(self,inp,oup):
        self.inp = inp
        self.oup = oup
    
    def summarize(self):
        self.model.summary()

    def train(self, new_inp=None, new_oup=None, verbose_=0, epochs_=75):
        if new_inp == None or new_oup == None:
            self.history = self.model.fit(self.inp, self.oup, epochs=epochs_, verbose=verbose_, shuffle=True, callbacks=[self.history])
        else:
            self.history = self.model.fit(new_inp, new_oup, epochs=epochs_, verbose=verbose_, shuffle=True, callbacks=[self.history])
        return self.history

    def drawHist(self, factor):
        hist = self.history.history[factor]
        _, hist_ax = pl.subplots()
        hist_ax.set_yscale("log")
        hist_ax.plot(hist)

    def replaceModel(self, model):
        self.model = model
        self.model.compile(optimizer='adam', loss='mse')
