import numpy as np 
import argparse

import os
from os import listdir
from os.path import isfile, join

from utils import RMSD

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.decomposition import PCA
from scipy import special


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

hex_list = ['#ffffff', '#82B182']
hex_list_cailloux = ['#ffffff', '#bfa804']
hex_list_sol = ['#ffffff', '#b94603']

def legendre(n,X) :
    for m in range(n+1):
        if(m==0):
          res=special.lpmv(m,n,X)
          res=res.reshape(1,res.shape[0])
        else:
          res=np.vstack((res,special.lpmv(m,n,X)))
    return res


class K_Means_RMSD:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i+25].T
            #print(self.centroids[i].shape)

        for i in range(self.max_iter):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for featureset in data:
                #distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                #print(self.centroids)
                distances = [RMSD(featureset.T,self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset.T)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data.T-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

def voxels_view_3D(data, i, save_path,name,treshold=1.):

    print(data.shape)
    fig = plt.figure(i)
    ax  = fig.gca(projection='3d')

    colors = get_continuous_cmap(hex_list)(data)

    ax.dist = 10.
    ax.axis = ('equal')
    ax.voxels(data, facecolors=colors)#,edgecolor='k')
    ax.grid(False)
    ax.set_axis_off()

    plt.savefig(os.path.join(save_path,name))
    plt.close(fig)          

if __name__ == '__main__':

    ############################ PROCESS ARGUMENTS ##################################

    parser = argparse.ArgumentParser(description="Train process of a Fake News Classifier")

    # Load arguments
    parser.add_argument("--load_path","-lp", type=str, default='../decompositions/3D/L18',
        help="Path to a directory with Spherical decompositions")
    parser.add_argument("--save_path","-sp", type=str, default='../data/class_3DForm_50/',
        help="Path to a directory where to save kmeans group")

    parser.add_argument("--load_name","-ln", type=str, default='mic')

    parser.add_argument("--kgroup","-k", type=int, default=7,
        help="K parameter for Kmeans clustering")

    args = parser.parse_args()

    LOAD_PATH = args.load_path
    SAVE_PATH = args.save_path
    
    decompositions_dir = np.sort([f for f in listdir(LOAD_PATH)])

    all_decompositions = {'error': {k:0. for k in decompositions_dir}, 'CC': {k:None for k in decompositions_dir}, 'idx': {k:0 for k in decompositions_dir},'npy': {k:0 for k in decompositions_dir}}
    all_errors=[]
    all_CC=[]
    minus=0
    for i,d in enumerate(decompositions_dir):
        if d in [".DS_Store","torus"]:
            minus+=1
            pass
        else:
            d_path = os.path.join(LOAD_PATH,d)
            try:
                error=np.load(os.path.join(d_path,'error.npy'))
                print("error = "+str(error*100.)+"%")

                CC = np.load(os.path.join(d_path,'CC.npy'))
                all_decompositions['error'][d] = error
                all_decompositions['CC'][d]    = CC
                all_decompositions['idx'][d]   = i-minus
                all_decompositions['npy'][d]   = d_path
                all_errors.append(error)
                all_CC.append(CC)
                print(d_path)
            except FileNotFoundError:
                print('FILENOTFOUND',d_path)
                minus+=1
                pass
                
    K= args.kgroup
    kmeans = K_Means_RMSD(k=K)
    kmeans.fit(all_CC)
    
    for i in range(K): os.makedirs('{}/class{i}'.format(SAVE_PATH,i),exist_ok=True)

    all_classifications = {k:None for k in decompositions_dir}
    all_labels = []
    for i,d in enumerate(decompositions_dir):
        cc = all_decompositions['CC'][d]
        all_classifications[d] = kmeans.predict(cc)
        all_labels.append(all_classifications[d])
        print(d,all_classifications[d])
        mesh_form = np.load(os.path.join(all_decompositions['npy'][d],'{}.npy'.format(args.load_name)),allow_pickle=True)
        name = all_decompositions['npy'][d].split('/')[-1]
        voxels_view_3D(mesh_form,i,'{}/class{}'.format(SAVE_PATH,all_classifications[d]),f'{name}')


