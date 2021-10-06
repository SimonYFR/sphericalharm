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

from utils.utils import scatter_3D, cut_to_txt
import time

from modules.Decomposer import SphericalHarmonicsDecomposer
from modules.Form import Form3D, Form2D
from modules.Loader import TXTFormLoader, RAWFormLoader
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

	############################ PROCESS ARGUMENTS ##################################

	parser = argparse.ArgumentParser(description="Train process of a Fake News Classifier")

	# Data arguments
	parser.add_argument("--object_path","-op", type=str, default='data/3DSyntheticForm/raw/defect10.raw',
	        help="Path to a 3D Form .txt file or .raw")

	# Cut generation arguments
	parser.add_argument("--nb_cut","-nc", type=int, default=1,
	        help="Number of 2D planar cut")

	# Saving arguments
	parser.add_argument("--save_path","-sp", type=str, default='data/2DSyntheticCut/txt/defect10/',
	        help="Directory path to save 2D planar cut")

	# Consol informations
	parser.add_argument("--force","-f", action='store_true',
	          help="Force to y WARNING")
	parser.add_argument("--verbose","-v", action='store_true',
	          help="Display consol informations")


	args = parser.parse_args()

	OBJ_PATH    = args.object_path
	SAVE_PATH   = args.save_path
	NB_CUT      = args.nb_cut
	VERB        = args.verbose
	FORCE       = args.force

	print(f"\n----- LOAD PATH -----")
	print(f"3D Object path (load) : {OBJ_PATH}")

	print(f"\n----- SAVE PATH -----")
	print(f"2D Planar cut path (save) : {SAVE_PATH}")

	print(f"\n----- CUT GENERATION -----")
	print(f"Number of cut (not used) : {NB_CUT}")

	try:
		os.makedirs(SAVE_PATH)
	except FileExistsError:
		if not FORCE : 
			answer = input("WARNING : The save path already exists, do you want to use it ? (y/n)\n")
			if answer == 'y':
				os.makedirs(SAVE_PATH,exist_ok=True)
			else:
				print("Exiting")
				exit()
		else:
			os.makedirs(SAVE_PATH,exist_ok=True)



	############################ LOADING 3D OBJECT #######################################

	loader_selector = {
	'raw' : RAWFormLoader,
	'txt' : TXTFormLoader
	}

	file_type = OBJ_PATH.split('.')[-1]
	file_name = OBJ_PATH.split('.')[-2]

	if file_type not in loader_selector.keys(): 
		print("WARNING : File type is not implemented (.raw or .txt)")
		print("Exiting")
		exit()

	ratio = 1.5
	loader = loader_selector.get(file_type)(OBJ_PATH)
	form,scales   = loader.load_form(ratio)

	DIM   = loader.dim
	if DIM == 2:
		print("WARNING : Performing cut generation must be on a 3D Object")
		print("Exiting")
		exit()

	OBJECT = Form3D(form, scales)

	print(f"\n----- 3D Object caracteristics -----")
	print(f"  * Number of voxels : {OBJECT.n_voxel}")
	print(f"  * Cube shape (L) : {OBJECT.shape}")
	print(f"  * Barycenter : {OBJECT.barycenter}")


	############################ GENERATION 2D CUT #######################################

	all_cuts    = OBJECT.get_all_2D_cut()
	
	train_cut, val_test_cut = train_test_split(all_cuts, test_size=0.3, random_state=42)
	val_cut, test_cut       = train_test_split(val_test_cut, test_size=0.3, random_state=42)

	set_selector = {'train':train_cut,
					'val':val_cut,
					'test':test_cut
					}

	max_Lmax    = 12

	SHD = SphericalHarmonicsDecomposer(OBJECT)
	SHD.fit(Lmax=max_Lmax,nphi=25,ntheta=25,points_selection='surface')
	
	spectre = np.vstack((np.real(SHD.cx),np.real(SHD.cy),np.real(SHD.cz),
	 					 np.imag(SHD.cx),np.imag(SHD.cy),np.imag(SHD.cz)))
	#spectre = np.vstack((np.real(SHD.cx),np.imag(SHD.cx),
	#					 np.real(SHD.cy),np.imag(SHD.cy),
	#					 np.real(SHD.cz),np.imag(SHD.cz)))

	for phase in ['train','val','test']:
		
		cuts = set_selector.get(phase)

		all_cut_spectre = []
		for axis,plane_idx in cuts:

			Lmax = max_Lmax

			cut  = OBJECT.get_2D_cut(axis,plane_idx)

			# ************** Compute Spherical Decomposition ************** #

			if (Lmax + 1)**2 >= cut.size_b:
				if np.sqrt(cut.size_b) % 1 == 0.:
					Lmax = int(np.sqrt(cut.size_b) - 2)
				else:
					Lmax = int(np.sqrt(cut.size_b) - 1)
			
			last_part = np.zeros((max_Lmax+1)**2 - (Lmax+1)**2)

			SHD = SphericalHarmonicsDecomposer(cut)
			SHD.fit(Lmax=Lmax,nphi=25,ntheta=25,points_selection='surface')

			#CC = np.vstack( (SHD.cx, SHD.cy) )
			# CC = np.vstack((np.hstack((np.real(SHD.cx),last_part)),
			# 			  	np.hstack((np.real(SHD.cy),last_part)),
			# 			  	np.hstack((np.imag(SHD.cx),last_part)),mes
			# 			  	np.hstack((np.imag(SHD.cy),last_part))))
			CC = np.vstack((np.hstack((np.real(SHD.cx),last_part)),
						  	np.hstack((np.real(SHD.cy),last_part)),
						  	np.hstack((np.imag(SHD.cx),last_part)),
						  	np.hstack((np.imag(SHD.cy),last_part))))			

			all_cut_spectre.append(CC)

		np.save(os.path.join(SAVE_PATH,f'all_CC_{phase}'), np.array([spectre,all_cut_spectre],dtype=object))


