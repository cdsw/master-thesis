import ipfshttpclient as ip

class IPFS:
    def __init__(self, id_):
        self.id_ = id_
        self.cli = ip.connect()
        self.id = self.cli.id()['ID']
    
    def sendToIPFS(self, file_loc):
        res = self.cli.add(file_loc)
        hash_ = res['Hash']
        del res
        return hash_
    
    def getFromIPFS(self, hash_, destination):
        self.cli.get(hash_, destination)

    def close(self):
        self.cli.close()