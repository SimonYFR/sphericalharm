import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os

def scatter_3D(save_path,name,title,x,y,z,xl='x',yl='y',zl='z'):
	fig = plt.figure(name)
	ax = fig.gca(projection='3d')
	ax.scatter3D(x, y, z,cmap='cool')
	ax.set_xlabel(xl)
	ax.set_ylabel(yl)
	ax.set_zlabel(zl)
	plt.title(title)
	plt.savefig(save_path,bbox_inches='tight')

def scatter_2D(save_path,name,title,x,y):
	fig = plt.figure(name)
	for ix,iy in zip(x,y):
		plt.scatter(ix, iy)
	plt.gca().invert_yaxis()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.title(title)
	plt.savefig(save_path,bbox_inches='tight')

def voxels_view_3D(m, i, l, save_path):
    fig = plt.figure(i)
    ax  = fig.gca(projection='3d')

    x,y,z = np.mgrid[:l,:l,:l]

    data   = m * np.indices((l,l,l))[0] * np.indices((l,l,l))[1] * np.indices((l,l,l))[2]
    colors = plt.cm.cool(data)

    ax.dist = 10.
    ax.axis = ('equal')
    ax.voxels(m, facecolors=colors)
    ax.grid(False)
    ax.set_axis_off()

    plt.savefig(os.path.join(save_path,'MIC'+str(i)),bbox_inches='tight')
    plt.close()

def voxels_view_2D(m, i, l, save_path):
	fig = plt.figure(i)
	ax  = fig.add_subplot(1, 1, 1)

	ax.axis = ('equal')
	ax.imshow(m)
	ax.grid(False)
	ax.set_axis_off()

	plt.savefig(os.path.join(save_path,'MIC'+str(i)))
	plt.close()

def imshow_2D(save_path,save_name,title,m):
	fig = plt.figure(i)
	ax  = fig.add_subplot(1, 1, 1)

	ax.imshow(m,cmap='cool')
	ax.title(title)

	plt.savefig(os.path.join(save_path,save_name),bbox_inches='tight')
	plt.close()







def cut_to_txt(form,file_path='test.txt'):
	voxels_idx = form.indexes.T
	with open(file_path, 'w') as f:
		for idx in voxels_idx: f.write(','.join(str(i) for i in idx) + '\n')







def RMSD(CC1,CC2):
    coef       = 1 / (4 * np.pi)
    l1_2, dim1 = CC1.shape
    l2_2, dim2 = CC2.shape

    assert(dim1 == dim2)

    Lmax = int(np.sqrt(min(l1_2, l2_2)) - 1)

    sum=0.
    for l in range(Lmax+1):
        for m in range(-l,l+1):
            j    = l**2 + l + m
            sum += np.linalg.norm(CC1[j] - CC2[j])**2

    return np.sqrt(coef * sum)


