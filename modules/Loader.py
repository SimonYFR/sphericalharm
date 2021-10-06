import numpy as np

class TXTFormLoader(object):

	def __init__(self,obj_path):
		self.obj_path = obj_path
		self.dim      = None
		self.data     = self.load_data()

	def load_data(self):

		data_x, data_y, data_z = [], [], []

		with open(self.obj_path,'r') as f:
			for i, line in enumerate(f):
				line_split = line.split(',')
				dim = len(line_split)

				data_x.append(int(line_split[0]))
				data_y.append(int(line_split[1]))
				if dim == 3: data_z.append(int(line_split[2]))

		if data_z:
			self.dim = 3
			return [data_x, data_y, data_z]
		else:
			self.dim = 2
			return [data_x, data_y]

	def load_form(self,ratio):

		form_selector = {
			2: self.load_form_2D,
			3: self.load_form_3D
		}

		return form_selector.get(self.dim)(ratio)

	def load_form_2D(self,square_ratio):

		if square_ratio < 1.0: square_ratio = 1.0

		get_data = lambda i: self.data[i] - np.min(self.data[i])
		x, y = get_data(0), get_data(1)
		
		get_laps = lambda x: np.max(x) - np.min(x)
		laps_x, laps_y = get_laps(x), get_laps(y)

		square_size = np.max( [laps_x, laps_y] )
		l         = int( square_ratio * square_size )
		form      = np.zeros( (l,l) )

		get_scale = lambda laps: int( (l - laps) / 2)
		scale_x, scale_y = get_scale(laps_x), get_scale(laps_y)

		for i_x, i_y in zip(x, y):
			i_x += scale_x
			i_y += scale_y
			form[i_x][i_y] = 1

		return form, [scale_x,scale_y]

	def load_form_3D(self,cube_ratio):

		if cube_ratio < 1.0: cube_ratio = 1.0

		get_data = lambda i: self.data[i] - np.min(self.data[i])
		x, y, z = get_data(0), get_data(1), get_data(2)
		
		get_laps = lambda x: np.max(x) - np.min(x)
		laps_x, laps_y, laps_z = get_laps(x), get_laps(y), get_laps(z)

		cube_size = np.max( [laps_x, laps_y, laps_z] )
		l         = int( cube_ratio * cube_size )
		form      = np.zeros( (l,l,l) )

		get_scale = lambda laps: int( (l - laps) / 2)
		scale_x, scale_y, scale_z = get_scale(laps_x), get_scale(laps_y), get_scale(laps_z)

		for i_x, i_y, i_z in zip(x, y, z):
			i_x += scale_x
			i_y += scale_y
			i_z += scale_z
			form[i_x][i_y][i_z] = 1

		return form, [scale_x,scale_y,scale_z]


class RAWFormLoader(object):

	def __init__(self,obj_path):
		self.obj_path = obj_path
		self.dim      = 3
		self.data     = self.load_data()

	def load_data(self):

		l=150

		raw_data = np.empty(l**3, np.uint8)

		with open(self.obj_path,'rb') as f:
			raw_data.data[:] = f.read()

		raw_data  = raw_data.reshape((l, l, l))
		data = np.where(raw_data == 1)

		return [data[0], data[1], data[2]]

	def load_form(self,ratio):

		form_selector = {
			3: self.load_form_3D(ratio)
		}

		return form_selector.get(self.dim)

	def load_form_3D(self,cube_ratio):

		if cube_ratio < 1.0: cube_ratio = 1.0

		get_data = lambda i: self.data[i] - np.min(self.data[i])
		x, y, z = get_data(0), get_data(1), get_data(2)
		
		get_laps = lambda x: np.max(x) - np.min(x)
		laps_x, laps_y, laps_z = get_laps(x), get_laps(y), get_laps(z)

		cube_size = np.max( [laps_x, laps_y, laps_z] )
		l         = int( cube_ratio * cube_size )
		form      = np.zeros( (l,l,l) )

		get_scale = lambda laps: int( (l - laps) / 2)
		scale_x, scale_y, scale_z = get_scale(laps_x), get_scale(laps_y), get_scale(laps_z)

		for i_x, i_y, i_z in zip(x, y, z):
			i_x += scale_x
			i_y += scale_y
			i_z += scale_z
			form[i_x][i_y][i_z] = 1

		return form, [scale_x,scale_y,scale_z]
