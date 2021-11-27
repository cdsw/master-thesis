import ipfshttpclient as ip
import os

class IPFSN:
    def __init__(self, id_):
        self.id_ = id_
        self.cli = ip.connect()
        self.id = self.cli.id()['ID']
    
    def submit(self, file_loc):
        res = self.cli.add(file_loc)
        hash_ = res['Hash']
        del res
        return hash_
    
    def fetch(self, hash_, destination):
        self.cli.get(hash_, destination)
        # later refer to the destination to access the model

    def close(self):
        self.cli.close()