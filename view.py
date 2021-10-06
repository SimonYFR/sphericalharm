import sys
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D  
np.set_printoptions(threshold=np.inf)
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from scipy import ndimage, misc
from scipy.signal import resample_poly
import scipy.interpolate as SP

import os
import argparse
import yaml
import shutil

from utils.utils import voxels_view_3D, voxels_view_2D

if __name__ == '__main__':

    ############################ PROCESS ARGUMENTS ##################################

    parser = argparse.ArgumentParser(description="Train process of a Fake News Classifier")

    # Load arguments
    parser.add_argument("--load_path","-lp", type=str, default='decompositions/3D/default',
        help="Path to a 3D Spherical decompositions")
    parser.add_argument("--load_name","-ln", type=str, default='mic',
        help="Name of decompositions (default : mic")

    # Saving arguments
    parser.add_argument("--save_path","-sp", type=str, default='reconstructions/3D/default',
        help="Directory path to save reconstruction figures")
    parser.add_argument("--save","-s", action="store_true",
        help="Name of decompositions (default : mic")

    # Consol informations
    parser.add_argument("--force","-f", action='store_true',
              help="Force to y WARNING")
    parser.add_argument("--verbose","-v", action='store_true',
              help="Display and save plots")

    args = parser.parse_args()

    LOAD_PATH = args.load_path
    LOAD_NAME = args.load_name
    SAVE_PATH = args.save_path
    SAVE      = args.save
    VERB      = args.verbose
    FORCE     = args.force

    print(f"\n----- LOAD PATH -----")
    print(f"3D Decompositions path (load) : {LOAD_PATH}")

    print(f"\n----- SAVE PATH -----")
    print(f"Save (bool) : {SAVE}")
    print(f"Figures path (save) : {SAVE_PATH}")

    try:
        os.makedirs(SAVE_PATH)
    except FileExistsError:
        if not FORCE:
            answer = input("WARNING : The save path already exists, do you want to use it ? (y/n)\n")
            if answer == 'y': os.makedirs(SAVE_PATH,exist_ok=True)
            else:
              print("Exiting")
              exit()
        else:
            os.makedirs(SAVE_PATH,exist_ok=True)



    ############################ CONFIGURATION EXTRACTION ##################################

    try:
        CONFIG_PATH = os.path.join(LOAD_PATH,'3Dconfig.yaml')
        # Reading configuration of the process
        with open(CONFIG_PATH, "r") as f:
          config = yaml.load(f, Loader=yaml.Loader)

    except FileNotFoundError:
        CONFIG_PATH = os.path.join(LOAD_PATH,'2Dconfig.yaml')
        # Reading configuration of the process
        with open(CONFIG_PATH, "r") as f:
          config = yaml.load(f, Loader=yaml.Loader)

    # Copy of the config in checkpoint
    shutil.copy2(CONFIG_PATH, SAVE_PATH)

    Lmax             = config["parameters"]["Lmax"]
    p_points         = config["parameters"]["ratio_points"]
    points_selection = config["parameters"]["selection"]
    L                = config["form"]["L"]

    DIM = len(L)

    print(f"\n----- Process Configuration -----")
    print(f"  * Mode : {Lmax}")
    print(f"  * Points selection : {points_selection}")
    print(f"  * Points ratio : {round(p_points * 100., 2)}%")

    ############################ CONFIGURATION EXTRACTION ##################################

    l     = np.max(L)
    mic   = np.load( os.path.join(LOAD_PATH,       LOAD_NAME + '.npy') )
    rmic  = np.load( os.path.join(LOAD_PATH,'r'  + LOAD_NAME + '.npy') )
    rrmic = np.load( os.path.join(LOAD_PATH,'rr' + LOAD_NAME + '.npy') )
    mic   = mic.astype(bool)
    rmic  = rmic.astype(bool)
    rrmic = rrmic.astype(bool)

    if DIM == 3:
        voxels_view_3D(mic   , 0, l, SAVE_PATH)
        voxels_view_3D(rmic  , 1, l, SAVE_PATH)
        voxels_view_3D(rrmic , 2, l, SAVE_PATH)

    if DIM == 2:
        voxels_view_2D(mic   , 0, l, SAVE_PATH)
        voxels_view_2D(rmic  , 1, l, SAVE_PATH)
        voxels_view_2D(rrmic , 2, l, SAVE_PATH)

    if VERB: plt.show()
