import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tqdm import tqdm
from multiprocessing import Pool

# Import map from txt #
map = np.genfromtxt('Maps/example_map.csv')


# Function that receives a position and a map and creates a random walk inside the map #
# The function returns the map with the random walk #

def random_walk(seed = 0):
        
        pos = np.array([29,15])

        np.random.seed(seed)
        
        random_walk = np.zeros(map.shape)
    
        # Create a random walk #

        for i in range(np.random.randint(1, 100)):

            x = np.random.randint(-1, 2)
            y = np.random.randint(-1, 2)

            # Only allow movement if the new position is inside the map and the next position is 1 in map #

            j = 0
            while j < 4:

                if pos[0] + x >= 0 and pos[0] + x < map.shape[0] and pos[1] + y >= 0 and pos[1] + y < map.shape[1] and map[pos[0] + x, pos[1] + y] == 1:
                    pos[0] += x
                    pos[1] += y
                    # Add 1 to the position of the random walk #
                    random_walk[pos[0], pos[1]] = 1
                    j += 1
                else:
                    # If the new position is not inside the map, change the direction #
                    x = np.random.randint(-1, 2)
                    y = np.random.randint(-1, 2)
                

        # Dilate the random walk using a circular mask of radius r, using convolve#
        r = 2
        mask = np.zeros((2 * r + 1, 2 * r + 1))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - r) ** 2 + (j - r) ** 2 <= r ** 2:
                    mask[i, j] = 1

        random_walk = np.clip(convolve2d(random_walk, mask, mode='same'), 0, 1)

        return random_walk

# Show examples images of random walk in a 3x3 grid #

fig, ax = plt.subplots(3, 3, figsize=(10, 10))

for i in range(3):
    for j in range(3):
        image = random_walk()
        ax[i, j].imshow(image)
        ax[i, j].axis('off')

plt.imshow(image)
plt.show()


if __name__ == '__main__':
    # Create a list of seeds to be used in the multiprocessing #
    N = 10000
    seeds = np.random.randint(0, 100000, N)
    results = []

    # execute in parallel create_map function N times #
    with Pool(4) as p:
        for data in tqdm(p.imap_unordered(random_walk, seeds), total=N):
            results.append(data)

    #Convert data to np array and save it as a npy file in Data
    data_stacked = np.array(results)
    np.save('DatasetConstruction/Data/SensingMasks.npy', results)
