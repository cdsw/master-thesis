from T4MLTraining import *
from tensorflow.python.keras.models import clone_model
from copy import deepcopy
from math import log10

class Client:
    def __init__(self, id_, inp_train, out_train, inp_test, out_test):
        self.id_ = id_
        self.inp_train = inp_train
        self.out_train = out_train
        self.inp_test = inp_test
        self.out_test = out_test
        self.deviation = 1

    def getWeight(self):
        return sum(flatten(self.inp_train.tolist()[0]))
        
    def extract(self):
        return self.inp_train, self.out_train, self.inp_test, self.out_test
    
    def setModel(self, model):
        model_copy = clone_model(model.model)
        #model_copy.build((None, len(self.inp_train[0]), 1))
        model_copy.compile(optimizer='adam', loss='mse')
        mw = model.model.get_weights()
        model_copy.set_weights(mw)
        del mw
        model = deepcopy(model)
        model.model = model_copy
        self.model = model
        self.model.setData(self.inp_train, self.out_train)

    def train(self, draw_loss=False):
        ts = time.time()
        w = self.getWeight()
        ep = int((log10(w))*40) + 1
        print(':', w, ep,end= ' -- ')
        self.model.train(epochs_=ep)
        te = time.time()
        print("Cli# {0}: {1:.2f} sec".format(self.id_, te - ts), end = '; ')
        if draw_loss == True:
            self.model.drawLoss()

    def extractModel(self):
        return self.model
    
    def predict(self, frame_in, frame_out, verbose_=False):
        predictor = Prediction(self.inp_test,self.out_test,self.model, frame_in, frame_out)
        predictor.predict()
        predictor.summary("",verbose_)
        self.deviation = predictor.in_percent / 100
        return self.deviation

class Distributor:
    def __init__(self, dat, company, location, div_company=True, div_location=False, bin_='40T', frame_out=2, ratio=0.8):
        self.dat = dat
        self.company = company
        self.location = location
        self.divide_by_company = div_company
        self.divide_by_location = div_location # NOT YET IMPLEMENTED
        self.clients = []
        
        self.bin_=bin_
        self.frame_out = frame_out
        self.ratio = ratio

    def divByCompany(self):
        for c in self.company:
            tr = DataPrep(self.dat,[c],self.location,[c],self.location,self.bin_,self.frame_out,self.ratio)
            tr.setup()

            inp_train, out_train, inp_test, out_test = tr.extract()
            cli = Client(str(c), inp_train, out_train, inp_test, out_test)
            self.clients.append(cli)
    
    def divCustom(self, pairs):
        for company,location in pairs:
            tr = DataPrep(self.dat,company,location,company,location,self.bin_,self.frame_out,self.ratio)
            tr.setup()

            inp_train, out_train, inp_test, out_test = tr.extract()
            cli = Client(str(company) + '/' + str(location), inp_train, out_train, inp_test, out_test)
            self.clients.append(cli)

    def divBoth(self):
        for c in self.company:
            for l in self.location:
                tr = DataPrep(self.dat,[c],[l],[c],[l],self.bin_,self.frame_out,self.ratio)
                tr.setup()

                inp_train, out_train, inp_test, out_test = tr.extract()
                cli = Client(str(c) + '/' + str(l), inp_train, out_train, inp_test, out_test)
                self.clients.append(cli)
                
    def extractClients(self):
        return self.clients

def weighting(model, wei):
    for i in range(len(model)):
        model[i] *= wei
    return model

def combine_models(m1, m2):
    m = deepcopy(m1)
    for i in range(len(m1)):
        m[i] += m2[i]
    return m

class Server:
    def __init__(self, clients, frame_in, frame_out, model_type, epochs_):
        self.clients = clients
        self.cli_models = []
        if model_type == 'CNNLSTM':
            self.model = CNNLSTM(frame_in, frame_out, epochs=epochs_)
        self.frame_in = frame_in
        self.frame_out = frame_out

    def askClients(self):
        for c in self.clients:
            c.train()
            m = c.extractModel()
            self.cli_models.append(m)      

    def aggregate(self, mode='amt'):
        multiplier = []
        cli_weights = []
        # get client demand size and weights
        for c in self.clients:
            if mode == 'amt':
                w = c.getWeight()
            elif mode == 'pre':
                w = c.predict(self.frame_in, self.frame_out, verbose_=0)
                try:
                    w = (1 / w)
                except ZeroDivisionError:
                    w = 1
            multiplier.append(w)
            cli_weights.append(c.model.model.get_weights())
        total_multiplier = sum(multiplier)

        # get proportions
        multiplier_proportion = []
        for w in multiplier:
            multiplier_proportion.append(w/total_multiplier)
        
        # averaging
        average = None
        for i in range(len(cli_weights)):
            weighted = weighting(cli_weights[i],multiplier_proportion[i])
            if average == None:
                average = weighted
            else:
                average = combine_models(average, weighted)

        # repacking in global model
        self.model.model.set_weights(average)
        self.model.model.build((None,self.frame_in,1))
        self.model.model.compile(optimizer='adam',loss='mse')

    def sendGlobalModel(self):
        for c in self.clients:
            c.setModel(self.model)

    def iterate(self, iters, mode='amt'):
        for i in range(iters):
            print("Iteration " + str(i + 1) + " of " + str(iters), end=' | ')
            tsg = time.time()
            self.sendGlobalModel() # loop start
            tsh = time.time()
            self.askClients()
            tsi = time.time()
            self.aggregate(mode)
            tsj = time.time()
            print(" | Glob: {0:.2f}s, Train: {1:.2f}s, Aggr: {2:.2f}s.".format(tsh-tsg, tsi-tsh, tsj-tsi))
        return self

