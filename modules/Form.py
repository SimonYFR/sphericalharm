import numpy as np

from scipy import ndimage as ndi

import matplotlib.pyplot as plt

class Form2D(object):

	def __init__(self,form,scales):
		self.form    = form
		self.shape   = self.form.shape
		self.n_voxel = np.sum(self.form)
		self.indexes  = np.array(np.where(self.form == 1))

		self.scale_x, self.scale_y = scales

		self.barycenter  = self.extract_barycenter()
		self.bx, self.by = self.extract_boundaries()

		self.size_b = len(self.bx)

	def extract_barycenter(self):

		form = self.form.astype(bool)

		D1  = ndi.distance_transform_edt(~form.copy())
		ND1 = ndi.distance_transform_edt(form.copy())

		inner_points 	  = D1-ND1
		mean_inner_points = np.mean(np.where(inner_points==np.min(inner_points)),axis=1)
		barycenter        = [round(mean_inner_points[1]),round(mean_inner_points[0])]

		return barycenter

	def centered_meshgrid(self):

		centered_mesh = lambda i: np.arange(0,self.shape[i]) - self.barycenter[i]
		
		X, Y = centered_mesh(0), centered_mesh(1)

		return np.meshgrid(X, Y)

	def extract_form(self):

		X, Y = self.centered_meshgrid()

		get_idx = lambda x: x[self.indexes[0], self.indexes[1]]

		x, y = get_idx(X), get_idx(Y)

		return x, y

	def extract_boundaries(self):

		X, Y = self.centered_meshgrid()

		m1 = abs(self.form[np.hstack((range(1,self.shape[0]),0)),:] - self.form)
		m2 = abs(self.form[:,np.hstack((range(1,self.shape[1]),0))] - self.form)

		# boundary x y z axis
		bx = np.hstack( (X[m1>0]    , X[m2>0]+0.5) )
		by = np.hstack( (Y[m1>0]+0.5, Y[m2>0]    ) )

		return bx, by

	def get_array_scales(self):
		return np.array([self.scale_x, self.scale_y])

	def get_array_form(self):
		return np.array(self.extract_form())

	def get_array_boundaries(self):
		return np.array([self.bx, self.by])

class Form3D(object):

	def __init__(self,form,scales):
		self.form    = form
		self.shape   = self.form.shape
		self.n_voxel = np.sum(self.form)
		self.indexes  = np.where(self.form == 1)

		self.scale_x, self.scale_y, self.scale_z = scales

		self.barycenter           = self.extract_barycenter()
		self.bx, self.by, self.bz = self.extract_boundaries()

		self.size_b = len(self.bx)

	def extract_barycenter(self):

		form = self.form.astype(bool)

		D1  = ndi.distance_transform_edt(~form.copy())
		ND1 = ndi.distance_transform_edt(form.copy())

		inner_points 	  = D1-ND1
		mean_inner_points = np.mean(np.where(inner_points==np.min(inner_points)),axis=1)
		barycenter        = [round(mean_inner_points[2]),round(mean_inner_points[1]),round(mean_inner_points[0])]

		return barycenter

	def centered_meshgrid(self):

		centered_mesh = lambda i: np.arange(0,self.shape[i]) - self.barycenter[i]

		X, Y, Z = centered_mesh(0), centered_mesh(1), centered_mesh(2)

		return np.meshgrid(X, Y, Z)

	def extract_form(self):

		X, Y, Z = self.centered_meshgrid()

		get_idx = lambda x: x[self.indexes[0], self.indexes[1], self.indexes[2]]  

		x, y, z = get_idx(X), get_idx(Y), get_idx(Z)

		return x, y, z

	def extract_boundaries(self):

		X, Y, Z = self.centered_meshgrid()

		m1 = abs(self.form[np.hstack((range(1,self.shape[0]),0)),:,:] - self.form)
		m2 = abs(self.form[:,np.hstack((range(1,self.shape[1]),0)),:] - self.form)
		m3 = abs(self.form[:,:,np.hstack((range(1,self.shape[2]),0))] - self.form)

		# boundary x y z axis
		bx = np.hstack( (X[m1>0]    , X[m2>0]+0.5, X[m3>0]    ) )
		by = np.hstack( (Y[m1>0]+0.5, Y[m2>0]    , Y[m3>0]    ) )
		bz = np.hstack( (Z[m1>0]    , Z[m2>0]    , Z[m3>0]+0.5) )

		return bx, by, bz

	def random_2D_cut(self):

		form_oriented   = self.form.copy()

		indexes = self.indexes

		axis     = np.random.randint(3)
		min_idx  = np.min(indexes[axis])
		max_idx  = np.max(indexes[axis])

		plane_idx = np.random.randint(min_idx, max_idx + 1)

		if axis == 0: cut = form_oriented[plane_idx, :, :].reshape( (self.shape[1],self.shape[2]) )
		if axis == 1: cut = form_oriented[:, plane_idx, :].reshape( (self.shape[0],self.shape[2]) )
		if axis == 2: cut = form_oriented[:, :, plane_idx].reshape( (self.shape[0],self.shape[1]) )

		return Form2D(cut, [self.scale_x,self.scale_y])

	def get_2D_cut(self,axis,plane_idx):

		if axis == 0: cut = self.form[plane_idx, :, :].reshape( (self.shape[1],self.shape[2]) )
		if axis == 1: cut = self.form[:, plane_idx, :].reshape( (self.shape[0],self.shape[2]) )
		if axis == 2: cut = self.form[:, :, plane_idx].reshape( (self.shape[0],self.shape[1]) )

		return Form2D(cut, [self.scale_x,self.scale_y])

	def get_all_2D_cut(self):

		indexes = self.indexes

		all_cuts = []
		for axis in [0,1,2]:

			min_idx  = np.min(indexes[axis])
			max_idx  = np.max(indexes[axis])

			for plane_idx in range(min_idx,max_idx+1):

				all_cuts.append([axis,plane_idx])

		return all_cuts

	def get_array_scales(self):
		return np.array([self.scale_x, self.scale_y, self.scale_z])

	def get_array_form(self):
		return np.array(self.extract_form())

	def get_array_boundaries(self):
		return np.array([self.bx, self.by, self. bz])


