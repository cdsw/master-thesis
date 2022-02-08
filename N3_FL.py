from N1_ML import *
from U1_IPFS import *
from tensorflow.python.keras.models import clone_model, load_model
from os import remove as rm
import pickle
from math import log

# FUNCTIONS MISC

def combine_models(m1, m2):
    m = deepcopy(m1)
    for i in range(len(m1)):
        m[i] += m2[i]
    return m

def multiply_scalar(model_, wei):
    model = deepcopy(model_)
    for i in range(len(model)):
        model[i] *= wei
    return model

def combine_ultimate(models, weights):
    average = None
    for i in range(len(models)):
        #print(models[i][0][0][0][0][0], 'weight', weights [i])
        weighted = multiply_scalar(models[i], weights[i])
        if average == None:
            average = weighted
        else:
            average = combine_models(average, weighted)
    #print(average[0][0][0][0][0], 'combine-weights')
    return average

# CLIENT

class Client:
    def __init__(self, id_):
        self.id_ = str(id_)
        self.hash = toHash(self.id_)
        self.round = 0 # <- useful for spacing out tests
        self.roundLimit = 1
        self.train_inp = None
        self.train_oup = None
        self.test_inp = None
        self.test_oup = None
        self.latest_quality = None
        self.contribution = None
        self.ipfs = IPFS(id_=self.id_)
        self.model = CNN()
        self.hashes = None

    def convertToClass(self, class_, init=0):
        self.__class__ = class_

    def getVolume(self):
        return len(self.train_inp)

    def setRoundLimit(self, limit):
        self.roundLimit = limit

    def train(self, loc, epoch_, round_):
        suf='temp.h5'
        qual_name = 'qual'
        portion_select_low = int(round_ / self.roundLimit * len(self.train_inp))
        portion_select_high = int((round_ + 1) / self.roundLimit * len(self.train_inp))
        inp = self.train_inp[portion_select_low:portion_select_high]
        oup = self.train_oup[portion_select_low:portion_select_high]
        self.model.setData(inp, oup)
        ts = time()
        self.latest_quality = self.model.trainW(str(self.id_), getSession(), epochs_=epoch_).history['accuracy'][-1]
        te = time()
        print("Cli# {0}: {1:.2f} sec, Q: {2:.4f}".format(self.id_, te - ts, self.latest_quality), end = '; ')
        # save model to file
        self.contribution = self.latest_quality * (self.getVolume() ** 0.5)
        saveToFile(str(self.contribution),loc+qual_name)
        self.model.model.save(loc+suf, save_format='h5')
        # save to IPFS
        mod_hash = self.ipfs.sendToIPFS(loc+suf)
        qual_hash = self.ipfs.sendToIPFS(loc+qual_name)
        rm(loc+suf)
        rm(loc+qual_name)
        self.hashes = [mod_hash, qual_hash]
        return mod_hash, qual_hash

    def evaluate(self):
        qual =  self.model.model.evaluate(self.test_inp, self.test_oup, verbose=2)      
        return qual

    def extractModel(self):
        return self.model
    
    def setData(self, train_inp_, train_oup_, test_inp_, test_oup_):
        self.train_inp = deepcopy(train_inp_)
        self.train_oup = deepcopy(train_oup_)
        self.test_inp = deepcopy(test_inp_)
        self.test_oup = deepcopy(test_oup_)

    def replaceWeights(self, weights):
        self.model.model.set_weights(weights)
        #self.model.model.build((None,self,36,1))
        self.model.model.build((None,self,28,28,1))
        self.model.model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])

# SERVER

class Server:
    def __init__(self, clients, epochs_, rounds_, model_dir):
        self.id_ = 'S'
        self.rounds = rounds_
        self.clients = clients
        self.model = CNN()
        self.model.setup()
        self.ipfs = IPFS(self.id_)
        self.cli_models = []
        self.cli_contrib = []
        self.epochs_ = epochs_
        self.test_inp = None
        self.test_oup = None
        self.model_dir = model_dir
    
    def setTestCase(self, tinp, tout):
        self.test_inp = deepcopy(tinp)
        self.test_oup = deepcopy(tout)
        
    def predict(self):
        self.model.model.evaluate(self.test_inp, self.test_oup, verbose=2)

    def askClients(self, round_):
        for c in self.clients:
            mod_hash, con_hash = c.train(self.model_dir, self.epochs_, round_)
            self.cli_models.append(mod_hash)
            self.cli_contrib.append(con_hash)

    def sendGlobalModel(self):
        suf = 'temp.h5'
        # save model to h5 as TEMP
        self.model.model.save(self.model_dir+suf, save_format='h5')
        # upload to IPFS
        glob_hash = self.ipfs.sendToIPFS(self.model_dir+suf)
        rm(self.model_dir+suf)
        # spread global model
        for c in self.clients:
            # client fetches the model to temp
            c.ipfs.getFromIPFS(glob_hash, self.model_dir)
            # load model
            model_weights = load_model(self.model_dir+glob_hash).get_weights()
            # replace model with the new model
            c.replaceWeights(model_weights)
            rm(self.model_dir+glob_hash)

    def iterate(self):
        for i in range(self.rounds):
            print("Iteration " + str(i + 1) + " of " + str(self.rounds), end=' | ')
            tsg = time()
            self.sendGlobalModel() # loop start
            tsh = time()
            self.askClients(i)
            tsi = time()
            self.aggregate()
            #self.model.setData(self.test_inp, self.test_oup)
            #self.model.trainW('Glob', getSession(), verbose_=2)
            tsj = time()
            print(" | Glob: {0:.2f}s, Train: {1:.2f}s, Aggr: {2:.2f}s.".format(tsh-tsg, tsi-tsh, tsj-tsi))
            self.cli_models = []
            self.cli_weights = []
            self.cli_contrib = []

        self.predict()

    def aggregate(self):
        # fetch models from hashes
        total_contribution = 0
        cli_weights = []
        cli_contribs = []
        cli_proportion = []

        for i in range(len(self.cli_models)):
            hash_ = self.cli_models[i]
            cont_ = self.cli_contrib[i]
            # import from IPFS hash to temps
            self.ipfs.getFromIPFS(hash_, self.model_dir)
            md = load_model(self.model_dir+hash_).get_weights()
            cli_weights.append(md)
            rm(self.model_dir+hash_)

            temp_save = "./temp/"
            self.ipfs.getFromIPFS(cont_, temp_save)
            ccb = float(readFromFile(temp_save + cont_))
            total_contribution += ccb
            cli_contribs.append(ccb)
            rm(temp_save + cont_)

        # get proportions
        for w in cli_contribs:
            cli_proportion.append(w/total_contribution)
        
        # averaging
        average = combine_ultimate(cli_weights, cli_proportion)

        # repacking in global model
        self.model.model.set_weights(average)
        #self.model.model.build((None,36,1))
        self.model.model.build((None,28,28,1))
        self.model.model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])

# POISONING FUNCTIONS

def shuffle(lst):
    lst_ = []
    for i in range(len(lst)):
        candidate = randint(-1,1)
        if lst[i][0] == 1 and lst[i][0] == candidate:
            candidate = 0
        lst_.append([candidate])
    return lst_

def poisonFL(client_):
    for i in range(len(client_.train_inp)):
        for j in range(len(client_.train_inp[i])):
            client_.train_inp[i][j] = shuffle(client_.train_inp[i][j])
    client_.train_oup = np.array(shuffle(client_.train_oup))
    return client_

# FL FUNCTIONS

def buildFL(train_inps_, train_oups_, test_inps_, test_oups_, test_inp_, test_oup_, epochs_, rounds_, model_dir):
    clients = []
    for i in range(len(train_inps_)):
        c = Client(i)
        c.model.setup()
        c.setData(train_inps_[i], train_oups_[i], test_inps_[i], test_oups_[i])
        c.setRoundLimit(rounds_)
        clients.append(c)
    server = Server(clients, epochs_, rounds_, model_dir)
    server.setTestCase(test_inp_, test_oup_)
    return server


def buildFLPoi(train_inps_, train_oups_, test_inps_, test_oups_, test_inp_, test_oup_, epochs_, rounds_, client_poi):
    clients = []
    for i in range(len(train_inps_)):
        c = Client(i)
        c.model.setup()
        c.setData(train_inps_[i], train_oups_[i], test_inps_[i], test_oups_[i])
        c.setRoundLimit(rounds_)
        if i in client_poi:
            c = poisonFL(c)
        clients.append(c)
    server = Server(clients, epochs_, rounds_, './models/')
    server.setTestCase(test_inp_, test_oup_)
    return server

def simulFL(client_poisoning, ep_it, train_inps, train_oups, test_inps, test_oups, test_inp, test_oup):
    for ep, it in ep_it:
        for i in range(len(client_poisoning)):
            print("EP = {:2d} IT = {:2d}, POI = ".format(ep, it) + str(client_poisoning[i]))
            server = buildFLPoi(train_inps, train_oups, test_inps, test_oups, test_inp, test_oup, ep, it, client_poisoning[i])
            server.iterate()