import numpy as np
import pdb
from numpy import linalg as LA
from scipy import spatial
from scipy import special
import sys
np.set_printoptions(threshold=np.inf)
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import os
import argparse
import yaml
import shutil

from utils.utils import scatter_3D, scatter_2D
import time

from modules.Decomposer import SphericalHarmonicsDecomposer
from modules.Form import Form3D, Form2D
from modules.Loader import TXTFormLoader, RAWFormLoader

if __name__ == '__main__':

  ############################ PROCESS ARGUMENTS ##################################

  parser = argparse.ArgumentParser(description="Train process of a Fake News Classifier")

  # Data arguments
  parser.add_argument("--object_path","-op", type=str, default='data/3DSyntheticForm/raw/defect10.raw',
            help="Path to a 3D Form .txt file or .raw")

  # Configuration arguments
  parser.add_argument("--config_path","-cp", type=str, default='config/3Dconfig.yaml',
            help="Training process configuration path (.yaml)")
  parser.add_argument("--test_bool",'-tb',action='store_true',
              help="Boolean dev value")

  # Saving arguments
  parser.add_argument("--save_path","-sp", type=str, default='decompositions/3D/default',
            help="Directory path to save decompositions (original,interpolation,spectre,construction)")
  parser.add_argument("--save_name","-sn", type=str, default='mic',
              help="Name of decompositions (default : mic")

  # Consol informations
  parser.add_argument("--force","-f", action='store_true',
              help="Force to y WARNING")
  parser.add_argument("--plot","-p", action='store_true',
              help="Display and save plots")
  parser.add_argument("--verbose","-v", action='store_true',
              help="Display consol informations")


  args = parser.parse_args()

  OBJ_PATH    = args.object_path
  CONFIG_PATH = args.config_path
  SAVE_PATH   = args.save_path
  SAVE_NAME   = args.save_name
  FIG_PATH    = os.path.join(SAVE_PATH,'plots')
  PLOT        = args.plot
  VERB        = args.verbose
  FORCE       = args.force

  print(f"\n----- LOAD PATH -----")
  print(f"3D Object path (load) : {OBJ_PATH}")
  print(f"Process configuration path (load)  : {CONFIG_PATH}")

  print(f"\n----- SAVE PATH -----")
  print(f"Decompositions path (save) : {SAVE_PATH}")
  print(f"Decompositions name (file) : {SAVE_NAME}")

  try:
    os.makedirs(SAVE_PATH)
    os.makedirs(FIG_PATH)
  except FileExistsError:
    if not FORCE : 
      answer = input("WARNING : The save path already exists, do you want to use it ? (y/n)\n")
      if answer == 'y':
        os.makedirs(SAVE_PATH,exist_ok=True)
        os.makedirs(FIG_PATH,exist_ok=True)
      else:
        print("Exiting")
        exit()
    else:
        os.makedirs(SAVE_PATH,exist_ok=True)
        os.makedirs(FIG_PATH,exist_ok=True)



  ############################ LOADING OBJECT #######################################

  loader_selector = {
    'raw' : RAWFormLoader,
    'txt' : TXTFormLoader
  }

  file_type = OBJ_PATH.split('.')[-1]

  if file_type not in loader_selector.keys(): 
    print("WARNING : File type is not implemented (.raw or .txt)")
    print("Exiting")
    exit()

  ratio = 1.2
  loader = loader_selector.get(file_type)(OBJ_PATH)
  form, scales   = loader.load_form(ratio=ratio)

  DIM   = loader.dim
  if DIM == 2: OBJECT = Form2D(form, scales)
  if DIM == 3: OBJECT = Form3D(form, scales)

  print(f"\n----- 3D Object caracteristics -----")
  print(f"  * Number of voxels : {OBJECT.n_voxel}")
  print(f"  * Cube shape (L) : {OBJECT.shape}")
  print(f"  * Barycenter : {OBJECT.barycenter}")
  print(f"  * Size boundary : {OBJECT.size_b}")



  ############################ CONFIGURATION EXTRACTION ##################################

  # Reading configuration of the process
  with open(CONFIG_PATH, "r") as f:
      config = yaml.load(f, Loader=yaml.Loader)
  # Copy of the config in checkpoint
  shutil.copy2(CONFIG_PATH, SAVE_PATH)

  Lmax             = config["parameters"]["Lmax"]
  p_points         = config["parameters"]["ratio_points"]
  points_selection = config["parameters"]["selection"]

  if points_selection == "surface":
    if (Lmax + 1)**2 >= OBJECT.size_b:
      Lmax = int(np.sqrt(OBJECT.size_b) - 1)
      config["parameters"]["Lmax"] = Lmax

  # Only useful for sphere and rand_surface points selection
  nphi   = int( np.sqrt(OBJECT.size_b * p_points) )
  ntheta = nphi
  nuni   = nphi * ntheta
  config["parameters"]["nphi"]   = nphi
  config["parameters"]["ntheta"] = ntheta
  config["form"]["L"]            = OBJECT.shape

  with open(os.path.join(SAVE_PATH,"3Dconfig.yaml"), "w") as f:
    yaml.dump(config, f)
    
  print(f"\n----- Process Configuration -----")
  print(f"  * Mode : {Lmax}")
  print(f"  * Points selection : {points_selection}")
  if points_selection in ['sphere','rand_surface']:
    print(f"  * Points ratio : {round(p_points * 100., 2)}%")
    print(f"  * Number of angle (polar)  : {ntheta}")
    print(f"  * Number of angle (azimuthal)  : {nphi}")

  ############################ PROCESS 3D OBJECT #######################################

  form = OBJECT.get_array_form()
  form_boundaries = OBJECT.get_array_boundaries()

  # ************** Compute Spherical Decomposition ************** #

  SHD = SphericalHarmonicsDecomposer(OBJECT, args.test_bool)
  SHD.fit(Lmax=Lmax,nphi=nphi,ntheta=ntheta,points_selection=points_selection)

  if PLOT and DIM == 2:
    scatter_2D(os.path.join(FIG_PATH,'original'),'original','Original',[form[0],SHD.bx], [form[1],SHD.by])
    scatter_2D(os.path.join(FIG_PATH,'boundary'),'boundary','Boundaries',[SHD.bx], [SHD.by])
    scatter_2D(os.path.join(FIG_PATH,'boundary_sphere'),'boundary_sphere','Boundaries Spherical Coordinates',[SHD.bsx], [SHD.bsy])
    scatter_2D(os.path.join(FIG_PATH,'boundary_selected'),'boundary_selected','Selected Boundaries',[SHD.xi], [SHD.yi])

  if PLOT and DIM == 3:
    selected_points = np.random.randint(0,len(SHD.bx),1000)
    scatter_3D(os.path.join(FIG_PATH,'original'),'original','Original',form[0], form[1], form[2])
    scatter_3D(os.path.join(FIG_PATH,'boundary'),'boundary','Boundaries',SHD.bx[selected_points], SHD.by[selected_points], SHD.bz[selected_points])
    scatter_3D(os.path.join(FIG_PATH,'boundary_sphere'),'boundary_sphere','Boundaries Spherical Coordinates',SHD.bsx[selected_points], SHD.bsy[selected_points], SHD.bsz[selected_points])
    scatter_3D(os.path.join(FIG_PATH,'boundary_selected'),'boundary_selected','Selected Boundaries',SHD.xi[selected_points], SHD.yi[selected_points], SHD.zi[selected_points])
  if VERB: plt.show()

  SHD.compute_full_reconstruction()
  if DIM == 2 : CC = np.vstack( (SHD.cx, SHD.cy) )
  if DIM == 3 : CC = np.vstack( (SHD.cx, SHD.cy, SHD.cz) )


  mic   = OBJECT.form
  rmic  = SHD.selected_points_reconstruction
  rrmic = SHD.form_reconstruction

  error=np.sum(abs(rrmic.ravel()-mic.ravel())) / float(np.sum(mic.ravel()))
  print("\n\nerror = "+str(error*100.)+"%\n\n")
  np.save(os.path.join(SAVE_PATH,'error'),error)

  np.save(os.path.join(SAVE_PATH, 'CC'            ), CC)
  np.save(os.path.join(SAVE_PATH,        SAVE_NAME), mic)
  np.save(os.path.join(SAVE_PATH, 'r'  + SAVE_NAME), rmic)
  np.save(os.path.join(SAVE_PATH, 'rr' + SAVE_NAME), rrmic)
