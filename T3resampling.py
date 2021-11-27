import pandas as pd

class Reader:
    def __init__(self, loc):
        self.dat = pd.read_csv(loc)
        self.subset = None

    def showStats(self, pickup=1, dropoff=0):
        print('Per-company count')
        print(self.dat['hvfhs_license_num'].value_counts())
        if pickup == 1:
            print('Per-area count: Pickup')
            print(self.dat['PULocationID'].value_counts()) 
        if dropoff == 1:
            print('Per-area count: Dropoff')
            print(self.dat['DOLocationID'].value_counts())

    def createSubset(self,company,location):
        self.subset = self.dat[self.dat.hvfhs_license_num.isin(company)]
        self.subset = self.dat[self.dat.PULocationID.isin(location)]
        self.subset.reset_index()
        self.subset = self.subset[['hvfhs_license_num','pickup_datetime','dropoff_datetime','PULocationID','DOLocationID']]

    def showSubsetStats(self):
        print(self.dat.groupby(['PULocationID','hvfhs_license_num']).size())
    
    def save(self,loc):
        self.subset.to_csv(loc)
        # Cleaning CSV file
        file1 = open(loc,"r")
        content = file1.read()
        check = content[0]
        file1.close()

        if check != 'n':
            file1 = open(loc,"w")
            add = 'n'
            file1.write(add + content)
            file1.close()

def main():
    # Loading data
    import os
    os.chdir('/home/cl/Documents/n-bafc/blockchain-afc/taxi')
    loc = 'trips.csv'
    dat = Reader(loc)
    # Show statistics
    dat.showStats()
    # Select only subset of companies and areas to reduce computation load
    company = [2,3,4,5]
    location = [mk for mk in range(9)]
    dat.createSubset(company,location)
    # Number of company-pickups
    dat.showSubsetStats()
    # Save into a simplified trip record
    loc = 'trips_simpler.csv'
    dat.save(loc)

#main()