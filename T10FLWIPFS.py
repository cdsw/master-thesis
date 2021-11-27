from T4MLTraining import *
from tensorflow.python.keras.models import clone_model, load_model
from copy import deepcopy
from random import randint
from U2IPFS import *

def weighting(model, wei):
    for i in range(len(model)):
        model[i] *= wei
    return model

def combine_models(m1, m2):
    m = deepcopy(m1)
    for i in range(len(m1)):
        m[i] += m2[i]
    return m

class ClientI:
    def __init__(self, id_, inp_train, out_train, inp_test, out_test):
        self.id_ = id_
        self.inp_train = inp_train
        self.out_train = out_train
        self.inp_test = inp_test
        self.out_test = out_test
        self.deviation = 1
        self.ipfs = IPFSN(id_=self.id_)
        self.hash = None
        self.model = CNNLSTM(36,2) # default model
        self.model.setData(self.inp_train, self.out_train)
        
    def getWeight(self):
        return sum(flatten(self.inp_train.tolist()[0]))
        
    def train(self, loc, epoch_, draw_loss=False):
        suf='temp.h5'
        ts = time.time()
        self.model.train(epochs_=epoch_)
        te = time.time()
        print("Cli# {0}: {1:.2f} sec".format(self.id_, te - ts), end = '; ')
        if draw_loss == True:
            self.model.drawLoss()
        # save model to file
        self.model.model.save(loc+suf, save_format='h5')
        # save to IPFS
        self.hash = self.ipfs.submit(loc+suf)
        os.remove(loc+suf)
        return self.hash

    def extractModel(self):
        return self.model
    
    def predict(self, frame_in, frame_out, verbose_=False):
        predictor = Prediction(self.inp_test,self.out_test,self.model, frame_in, frame_out)
        predictor.predict()
        predictor.summary("",verbose_)
        self.deviation = predictor.in_percent / 100
        return self.deviation

    def replaceWeights(self, weights):
        self.model.model.set_weights(weights)
        self.model.model.build((None,self,36,1))
        self.model.model.compile(optimizer='adam',loss='mse')

class ServerI:
    def __init__(self, clients, frame_in, frame_out, epochs_):
        self.id_ = 'S'
        self.clients = clients
        self.model = CNNLSTM(frame_in, frame_out, epochs=epochs_)
        self.frame_in = frame_in
        self.frame_out = frame_out
        self.ipfs = IPFSN(self.id_)
        self.cli_models = []
        self.loc = './models/'
        self.epochs_ = epochs_

    def askClients(self):
        for c in self.clients:
            # client trains itself
            hash_ = c.train(self.loc, self.epochs_)
            multi = c.getWeight()
            self.cli_models.append((hash_,multi))

    def sendGlobalModel(self):
        suf = 'temp.h5'
        # save model to h5 as TEMP
        self.model.model.save(self.loc+suf, save_format='h5')
        # upload to IPFS
        glob_hash = self.ipfs.submit(self.loc+suf)
        os.remove(self.loc+suf)
        # spread global model
        for c in self.clients:
            # client fetches the model to temp
            c.ipfs.fetch(glob_hash, self.loc)
            # load model
            model_weights = load_model(self.loc+glob_hash).get_weights()
            # replace model with the new model
            c.replaceWeights(model_weights)
            os.remove(self.loc+glob_hash)

    def iterate(self, iters):
        for i in range(iters):
            print("\nIteration " + str(i + 1) + " of " + str(iters), end=' | ')
            tsg = time.time()
            self.sendGlobalModel() # loop start
            tsh = time.time()
            self.askClients()
            tsi = time.time()
            self.aggregate()
            tsj = time.time()
            print(" | Glob: {0:.2f}s, Train: {1:.2f}s, Aggr: {2:.2f}s.".format(tsh-tsg, tsi-tsh, tsj-tsi))
        return self

    def aggregate(self):
        # fetch models from hashes
        cli_weights = []
        total_multiplier = 0
        multiplier = []
        multiplier_proportion = []

        for tup in self.cli_models:
            hash_, multi = tup[0], tup[1]
            multiplier.append(multi)
            total_multiplier += multi
            # import from IPFS hash to temps
            self.ipfs.fetch(hash_,self.loc)
            md = load_model(self.loc+hash_).get_weights()
            cli_weights.append(md)
            os.remove(self.loc+hash_)
            
        # get proportions
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
        self.model.model.build((None,36,1))
        self.model.model.compile(optimizer='adam',loss='mse')

class DistributorI:
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
            cli = ClientI(str(c), inp_train, out_train, inp_test, out_test)
            self.clients.append(cli)

    def divCustom(self, pairs):
        for company,location in pairs:
            tr = DataPrep(self.dat,company,location,company,location,self.bin_,self.frame_out,self.ratio)
            tr.setup()

            inp_train, out_train, inp_test, out_test = tr.extract()
            cli = ClientI(str(company) + '/' + str(location), inp_train, out_train, inp_test, out_test)
            self.clients.append(cli)

    def divBoth(self):
        for c in self.company:
            for l in self.location:
                tr = DataPrep(self.dat,[c],[l],[c],[l],self.bin_,self.frame_out,self.ratio)
                tr.setup()

                inp_train, out_train, inp_test, out_test = tr.extract()
                cli = ClientI(str(c) + '/' + str(l), inp_train, out_train, inp_test, out_test)
                self.clients.append(cli)

    def extractClients(self):
        return self.clients