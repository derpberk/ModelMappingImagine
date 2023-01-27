""" Pytorch Dataloader that loads the data from Data/AlgaeBloomData.npy and returns it as a tensor.
Each matrix from AlgaeBloomData.npy is an image of the datase. The label is the index of the image in the dataset. """

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap

class AlgaeBloomDataset(Dataset):

    def __init__(self, data_path = None, set = 'train'):
        """ If datapath is none, select from Data/AlgaeBloomData.npy"""
        self.data = np.load(data_path if data_path is not None else 'DatasetConstruction/Data/AlgaeBloomData.npy')
        # Load the sensing masks from the file #
        self.masks = np.load('DatasetConstruction/Data/SensingMasks.npy')

        if set == 'train':
            self.data = self.data[:int(len(self.data) * 0.8)]
            self.masks = self.masks[:int(len(self.masks) * 0.8)]
        else:
            self.data = self.data[int(len(self.data) * 0.8):]
            self.masks = self.masks[int(len(self.masks) * 0.8):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the data and the mask as two stacked tensors, and the index of the image #
        return torch.FloatTensor(np.stack((self.data[idx], self.masks[idx]))), idx

    def show(self, idx):
        plt.imshow(self.data[idx], cmap = algae_colormap)
        plt.show()


# Example of use

if __name__ == '__main__':

    dataset = AlgaeBloomDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    # show some data in a 3x3 grid 
    fig = plt.figure()
    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(dataset[i][0][0], cmap = algae_colormap, alpha=dataset[i][0][1])
    plt.show()

