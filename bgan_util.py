import os 
import glob
import numpy as np
import six
import cPickle
import tensorflow as tf
from scipy.ndimage import imread
from scipy.misc import imresize
import scipy.io as sio
from numpy import shape
import csv



def one_hot_encoded(class_numbers, num_classes):
    return np.eye(num_classes, dtype=float)[class_numbers]


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
        
        
def print_images(sampled_images, label, index, directory):
    import matplotlib as mpl
    mpl.use('Agg') # for server side
    import matplotlib.pyplot as plt

    def unnormalize(img, cdim):
        img_out = np.zeros_like(img)
        for i in xrange(cdim):
            img_out[:, :, i] = 255.* ((img[:, :, i] + 1.) / 2.0)
        img_out = img_out.astype(np.uint8)
        return img_out
        

    if type(sampled_images) == np.ndarray:
        N, h, w, cdim = sampled_images.shape
        idxs = np.random.choice(np.arange(N), size=(5,5), replace=False)
    else:
        sampled_imgs, sampled_probs = sampled_images
        sampled_images = sampled_imgs[sampled_probs.argsort()[::-1]]
        idxs = np.arange(5*5).reshape((5,5))
        N, h, w, cdim = sampled_images.shape

        
    fig, axarr = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            if cdim == 1:
                axarr[i, j].imshow(unnormalize(sampled_images[idxs[i, j]], cdim)[:, :, 0], cmap="gray")
            else:
                axarr[i, j].imshow(unnormalize(sampled_images[idxs[i, j]], cdim))
            axarr[i, j].axis('off')
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_aspect('equal')

    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(os.path.join(directory, "%s_%i.png" % (label, index)), bbox_inches='tight')
    plt.close("all")

    '''if "raw" not in label.lower():
        np.savez_compressed(os.path.join(directory, "samples_%s_%i.npz" % (label, index)),
                            samples=sampled_images)'''
                            

            
class FigPrinter():
    
    def __init__(self, subplot_args):
        import matplotlib as mpl
        mpl.use('Agg') # guarantee work on servers
        import matplotlib.pyplot as plt
        self.fig, self.ax_arr = plt.subplots(*subplot_args)
        
    def print_to_file(self, file_name, close_on_exit=True):
        import matplotlib as mpl
        mpl.use('probsAgg') # guarantee work on servers
        import matplotlib.pyplot as plt
        self.fig.savefig(file_name, bbox_inches='tight')
        if close_on_exit:
            plt.close("all")
        

class SynthDataset():
    
    def __init__(self, x_dim=100, num_clusters=10, seed=1234):
        
        np.random.seed(seed)
        
        self.x_dim = x_dim
        self.N = 10000
        self.true_z_dim = 2
        # generate synthetic data
        self.Xs = []
        for _ in xrange(num_clusters):
            cluster_mean = np.random.randn(self.true_z_dim) * 5 # to make them more spread
            A = np.random.randn(self.x_dim, self.true_z_dim) * 5
            X = np.dot(np.random.randn(self.N / num_clusters, self.true_z_dim) + cluster_mean,
                       A.T)
            self.Xs.append(X)
        X_raw = np.concatenate(self.Xs)
        self.X = (X_raw - X_raw.mean(0)) / (X_raw.std(0))
        print self.X.shape
        
        
    def next_batch(self, batch_size):

        rand_idx = np.random.choice(range(self.N), size=(batch_size,), replace=False)
        return self.X[rand_idx]



        
class MnistDataset():
    
    def __init__(self, data_dir):
        
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
        self.x_dim = [28, 28, 1]
        self.num_classes = 10
        self.dataset_size = self.mnist.train.images.shape[0]
        
    def next_batch(self, batch_size, class_id=None):
        
        if class_id is None:
            image_batch, labels = self.mnist.train.next_batch(batch_size)
            new_image_batch = np.array([(image_batch[n]*2. - 1.).reshape((28, 28, 1))
                                        for n in range(image_batch.shape[0])])

            return new_image_batch, labels



def load_wind():
    # data created on July 8th, WA 52 wind farms
    with open('wind/new.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)
    print(shape(rows))
    for x in range(52):
        train = rows[:104832, x].reshape(-1, 576)
        train = train / 16
        train=train*2.-1.

        # print(shape(train))
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    print("Shape of TrX", shape(trX))
    dataset_size=trX.shape[0]
    dim=[24, 24, 1]

    return dim, dataset_size, trX


def load_mixture():
    # data created on July 8th, WA 52 wind farms
    with open('wind/new.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)

    for x in range(52):
        train = rows[:104832, x].reshape(-1, 576)
        train = train / 16
        train=train*2.-1.

        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    trX=trX[0:5824]
    print("Shape of wind", shape(trX))

    with open('wind/solar_0722.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    rows=rows[:104832,:]
    print(shape(rows))
    trX2 = np.reshape(rows.T,(-1,576))
    print("Shape of solar", shape(trX2))
    m = np.ndarray.max(rows)
    print("maximum value of solar power", m)
    trX2 = trX2/m
    trX2 = trX2 * 2. - 1.
    trX = np.concatenate((trX, trX2), axis=0)
    print("Shape of all data", shape(trX))

    dataset_size=trX.shape[0]
    dim=[24, 24, 1]

    return dim, dataset_size, trX

def noise_mixture():
    # data created on July 8th, WA 52 wind farms
    with open('wind/new.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)

    for x in range(52):
        train = rows[:104832, x].reshape(-1, 576)
        train = train / 16
        train=train*2.-1.

        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    trX=trX[0:5824]
    print("Shape of wind", shape(trX))

    '''with open('wind/solar_0722.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    rows=rows[:104832,:]
    print(shape(rows))
    trX2 = np.reshape(rows.T,(-1,576))
    print("Shape of solar", shape(trX2))
    m = np.ndarray.max(rows)
    print("maximum value of solar power", m)
    trX2 = trX2/m
    trX2 = trX2 * 2. - 1.'''

    trX2=np.random.normal(loc=0.0, scale=1.0, size=(5824,576))
    trX2=np.clip(trX2, -1, 1)
    trX = np.concatenate((trX, trX2), axis=0)
    print("Shape of all data", shape(trX))

    dataset_size=trX.shape[0]
    dim=[24, 24, 1]

    return dim, dataset_size, trX