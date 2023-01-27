import sys
import matplotlib.pyplot as plt

sys.path.append('.')

from GroundTruthsModels.AlgaeBloomGroundTruth import algae_bloom
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# Function that creates a algae_bloom object, resets it, execute the step method a random number of times [1,100] and returns the map using read #

map_lake = np.genfromtxt('Maps/example_map.csv')

def create_map(seed):

        np.random.seed(seed)
        gt = algae_bloom(map_lake, dt=0.2)
        gt.reset()
    
        for i in range(np.random.randint(1,100)):
            gt.step()

    
        return gt.read()


if __name__ == '__main__':

    # Import the map of the lake from txt #


    """ Create a new AlgaeBloomGroundTruth object and return it """

    gt = algae_bloom(map_lake, dt=0.2)
    gt.reset()
    gt.render()

    # Create a list of seeds to be used in the multiprocessing #
    N = 10000
    seeds = np.random.randint(0, 100000, N)
    results = []
    # execute in parallel create_map function N times #
    with Pool(4) as p:
        for data in tqdm(p.imap_unordered(create_map, seeds), total=N):
            results.append(data)

    #Convert data to np array and save it as a npy file in Data
    data_stacked = np.array(results)
    np.save('DatasetConstruction/Data/AlgaeBloomData.npy', data_stacked)




