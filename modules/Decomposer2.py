import numpy as np
from numpy import linalg as LA
np.set_printoptions(threshold=np.inf)

from scipy import spatial
from scipy import special

import matplotlib.tri as tri

# default it is 2D form
# form boundaries as inputs (x,y,z)
class SphericalHarmonicsDecomposer(object):

  def __init__(self, form, spherical_dist=False):

    # Form caracteristics
    self.form            = form
    self.form_boundaries = form.get_array_boundaries()
    self.b               = form.barycenter
    self.L               = form.shape
    self.dim             = len(form.shape)
    
    # Form caracteristics extraction
    self.bx,  self.by,     self.bz   = self.extract_polar_coordinates()
    self.br,  self.btheta, self.bphi = self.compute_spherical_coordinates(spherical_dist)
    self.bsx, self.bsy,    self.bsz  = self.compute_unit_sphere_coordinates()

    # Empty initialisation
    self.n_points             = 0
    self.phi, self.theta      = None, None
    self.xi, self.yi, self.zi, self.ri = None, None, None, None

    self.sphere_points       = None
    self.sphere_indices      = None
    self.selected_points_idx = None

    self.Y                                      = None
    self.cx,   self.cy,   self.cz, self.cr     = None, None, None, None
    self.x_Yc, self.y_Yc, self.z_Yc             = None, None, None
    self.selected_points_reconstruction         = None

    self.RY                            = None
    self.rr                            = None
    self.x_RYc, self.y_RYc, self.z_RYc = None, None, None
    self.form_reconstruction           = None

  def show_params(self):
    print("L:",self.L)
    print("dim:",self.dim)
    print("n_points:",self.n_points)

  def legendre(self,n,X) :
    for m in range(n+1):
        if m==0:
          res = special.lpmv(m,n,X)
          res = res.reshape(1,res.shape[0])
        else:
          res = np.vstack((res,special.lpmv(m,n,X)))
    return res

  def extract_polar_coordinates(self):

    if self.dim == 2:
      return self.form_boundaries[0], self.form_boundaries[1], None
    
    if self.dim == 3:
      return self.form_boundaries

  def compute_spherical_coordinates(self,spherical_dist):

    if spherical_dist:
      return self.compute_spherical_coordinates_2()

    if self.dim == 2:
      br     = np.sqrt(self.bx**2+self.by**2)
      bphi   = np.arctan2(self.bx,self.by)
      btheta = np.ones(len(bphi))*(np.pi/2)

    if self.dim == 3:
      br     = np.sqrt(self.bx**2+self.by**2+self.bz**2)
      bphi   = np.arctan2(self.bx,self.by)
      btheta = np.arctan2(np.sqrt(self.bx**2+self.by**2),self.bz)
      #btheta = np.arctan2(br,self.bz)
      
    return br, btheta, bphi

  def compute_spherical_coordinates_2(self):

    if self.dim == 2:

      triang = tri.Triangulation(self.bx, self.by)
      list_vertex = {k:[] for k in range(len(self.bx))}
      for t in triang.edges:
        t1,t2 = t
        list_vertex[t1] = list_vertex[t1] + [t2]
        list_vertex[t2] = list_vertex[t2] + [t1]

      dist_vertex = {k:[] for k in range(len(self.bx))}
      for k in list_vertex.keys():
        list = list_vertex.get(k)
        vertex = np.array([self.bx[k],self.by[k]])
        for v in list: 
          dist = np.linalg.norm(vertex-np.array([self.bx[v],self.by[v]]))
          dist_vertex[k] += [dist]

      mask_edges = {k:[] for k in range(len(self.bx))}
      for k in mask_edges.keys():
        list = dist_vertex.get(k)
        argsort_list = np.argsort(list)
        edges = []
        for i in argsort_list[:2]:
          edges+=[[k,list_vertex[k][i]]] 
        mask_edges[k]+= edges

      edges = []
      for m in mask_edges.values():
        for e in m:
          e_start = [self.bx[e[0]], self.by[e[0]]]
          e_end = [self.bx[e[1]], self.by[e[1]]]
          edges.append([e_start,e_end])
      edges = np.array(edges)

      all_dist = []
      for s in range(len(self.bx)):
        init_node = 0
        final_node = s
        start_node = init_node
        dist = 0.
        path = 0
        seen = [start_node]
        while start_node != final_node:
          path +=1
          for k in [0,1]:

            list = dist_vertex[start_node]
            argsort_list = np.argsort(list)[:2]

            next_node = list_vertex[start_node][argsort_list[k]]
            #dist += dist_vertex[start_node][next_node]

            if list_vertex[start_node][argsort_list[k]] not in seen:
              dist += dist_vertex[start_node][argsort_list[k]]
              start_node = list_vertex[start_node][argsort_list[k]]
              break


          seen += [start_node]
        #print(seen)
        all_dist.append(dist)
        #print(f"{final_node} DIST",dist)

      #print(all_dist)
      #print(np.argmax(all_dist),all_dist[np.argmax(all_dist)])
      v = np.argmax(all_dist)
      complete_dist = all_dist[np.argmax(all_dist)] + dist_vertex[v][0]
      #print(complete_dist)

      normalization = ((2*np.pi)/complete_dist)
      #print(normalization)
      normalized_dist = (np.array(all_dist) * normalization) #- np.pi
      #print(normalized_dist)
      #print(np.max(normalized_dist))

      br=1.
      bphi = normalized_dist
      btheta = np.ones(len(bphi))*(np.pi/2)

      #print(bphi,btheta)

      return br, btheta, bphi

    if self.dim == 3:
      print("WARNING 3D DIST PARAMETRIZATION NOT DEVELOPPED")
      return self.compute_spherical_coordinates(False)
      
    return br, btheta, bphi

  def compute_unit_sphere_coordinates(self):

    br = 1. #self.br
    bsx = br*np.cos(self.bphi)*np.sin(self.btheta)
    bsy = br*np.sin(self.bphi)*np.sin(self.btheta)

    if self.dim == 2: bsz = br*np.arange(0,len(self.bx))
    if self.dim == 3: bsz = br*np.cos(self.btheta)

    return bsx, bsy, bsz

  def fit(self, Lmax, nphi=25, ntheta=25, points_selection = 'sphere',scale=1.):

    self.set_selected_points(nphi=nphi,ntheta=ntheta,points_selection=points_selection)
    self.fourrier_spherical_harmonics_decomposition(Lmax=Lmax)
    self.fourrier_spherical_harmonics_modelisation(Lmax=Lmax)
    self.compute_selected_points_reconstruction(scale=scale)
    self.compute_full_reconstruction(scale=scale)


  def set_selected_points(self,nphi=25,ntheta=25,points_selection='sphere'):
    
    pi = np.pi

    if self.dim == 2:
      ntheta = 1

    if points_selection == 'sphere':

      n_points = ntheta*nphi

      if self.dim == 2:
        dphi  = 2*pi/nphi
        phi   = np.arange((dphi-pi),0.5+2*pi,dphi)
        theta = np.array([np.pi/2])

      if self.dim == 3:
        dphi   = 2*pi/nphi
        dtheta = 1./ntheta
        phi    = np.arange((dphi-pi),0.5+2*pi,dphi)
        theta  = np.arccos(1-2*np.arange(dtheta,1.+dtheta,dtheta))

      # Set of (theta_i, phi_i) points uniformly sampled on the sphere
      sphere_points  = np.zeros((n_points,2))
      sphere_indices = np.zeros((n_points,2))

      k=0
      for i in range(ntheta):
        for j in range(nphi):
            sphere_points[k,:]  = np.array([theta[i], phi[j]])
            sphere_indices[k,:] = np.array([i, j])
            k += 1
      sphere_indices = sphere_indices.astype(int)

      # For each couple (theta_i, phi_i), find indices of closest point on the surface in that direction
      btheta = self.btheta.reshape((self.btheta.shape[0],1))
      bphi   = self.bphi.reshape((self.bphi.shape[0],1))
      
      surface_points = np.hstack((btheta,bphi))
      distance, selected_points_idx  = spatial.KDTree(surface_points).query(sphere_points)


    if points_selection == 'rand_surface' or points_selection == 'surface':

      if points_selection == 'rand_surface':
        n_points = ntheta * nphi
        selected_points_idx = np.random.randint(0, len(self.bx), n_points)

      if points_selection == 'surface': 
        n_points = len(self.bx)
        selected_points_idx = np.arange(0, n_points)

      phi   = self.bphi[selected_points_idx]
      theta = self.btheta[selected_points_idx]

      sphere_points  = np.zeros((n_points,2))
      sphere_indices = np.zeros((n_points,2))

      for k in range(n_points):
        sphere_points[k,:]  = np.array([theta[k], phi[k]])
        sphere_indices[k,:] = np.array([k, k])
      sphere_indices = sphere_indices.astype(int)

    # Values of x, y, z at closest point
    self.n_points            = n_points
    self.sphere_points       = sphere_points
    self.sphere_indices      = sphere_indices
    self.selected_points_idx = selected_points_idx

    self.phi   = phi
    self.theta = theta

    self.xi = self.bx[self.selected_points_idx]
    self.yi = self.by[self.selected_points_idx]
    self.ri = self.br[self.selected_points_idx]
    if self.dim == 3: self.zi = self.bz[self.selected_points_idx]

  def check_NaN(self,array,title):

    array_sum     = np.sum(array)
    array_has_nan = np.isnan(array_sum)

    if(array_has_nan):
      print(f'\n ****PROBLEM****\n****NaN in {title}****\n')


  def fourrier_spherical_harmonics_decomposition(self, Lmax):

    pi = np.pi

    if self.n_points == 0:
      print("WARNING : You must use set_selected_points before")

    assert( (Lmax+1)**2 < self.n_points)

    # build matrix y_ij (Shen et al. page 18)
    Y  = np.zeros((self.n_points, (1+Lmax)**2))*1.j
    # fill in all elements that require the computation of P^m_l(cos(theta)) with l fixed
    ct = np.cos(self.theta)

    #print("\n----- Spherical Harmonic Extraction -----")

    for l in range(Lmax+1):

      Pml = self.legendre(l,ct)

      #Pml(1,:), Pml(2,:), ..., Pml(l+1,:) = P^0_l(x), P^1_l(x), ..., P^l_l^(x)
      for m in range(-l,l+1):

        if (m<=0): 
          
          acc=1
          for k in range (l+m+1,l-m):
            acc=acc*(k)
          prod, sign = (-1)**m/acc, -1
          
        else     : prod, sign = 1, 1
        myPml=Pml[sign*m,:]*prod

        self.check_NaN(myPml,'myPml')

        j=l**2+l+m

        #print(f"  * l:{l}/{Lmax} | m:{m}",end='\r')

        if (m<=0): 
          acc=1
          for k in range (l+m+1,l-m):
            acc=acc*(k)
          nm = acc
        else     : 
          acc=1
          for k in range (l-m+1,l+m):
            acc=acc*(k)
          nm = 1/acc
        
        Y[:,j] = myPml.T * np.exp((1.j) * m * self.phi) * np.sqrt(np.complex((2 * l + 1) / (4 * pi) * nm))
  
    #Check if everything is good
    self.check_NaN(Y,'Y')

    #print("\n\n----- Least Square Resolution -----")

    def least_square(Y,x,title):
      #print(f"  * {title} dimension ...")
      return LA.lstsq(Y,x, rcond=None)[0]

    self.cx = least_square(Y, self.xi,'x')
    self.cy = least_square(Y, self.yi,'y')
    self.cr = least_square(Y, self.ri,'r')

    if self.dim == 3: self.cz = least_square(Y, self.zi,'z')

    self.Y = Y

  def fourrier_spherical_harmonics_modelisation(self, Lmax):

    pi = np.pi

    if self.dim == 2:
      N = [self.L[0],self.L[1]]
      x, y   = np.arange(0,N[0]), np.arange(0,N[1])
      rx, ry = np.meshgrid(x-self.b[0],y-self.b[1])
      rr     = np.sqrt(rx**2+ry**2)
      rphi   = np.arctan2(rx,ry).ravel()
      rtheta = np.ones(len(rphi))*(np.pi/2)

    if self.dim == 3:
      N = [self.L[0],self.L[1],self.L[2]]
      x, y, z    = np.arange(0,N[0]), np.arange(0,N[1]), np.arange(0,N[2]) 
      rx, ry, rz = np.meshgrid(x-self.b[0],y-self.b[1],z-self.b[2])
      rr         = np.sqrt(rx**2 + ry**2 + rz**2)
      rtheta     = np.arctan2((rx**2+ry**2)**0.5,rz).ravel()
      #rtheta     = np.arctan2(rr,rz).ravel()
      rphi       = np.arctan2(rx,ry).ravel()

    #print("\n----- Spherical Harmonic Reconstruction -----")
    # first store all Y-function that will be needed for the computations.
    RY  = np.zeros((np.asarray(N).prod(),(Lmax+1)**2))*(1.j)
    crt = np.cos(rtheta)

    for l in range(Lmax+1):
      Rml0 = self.legendre(l,crt)

      for m in range(-l,l+1):
        #print(f"  * l:{l}/{Lmax} | m:{m}",end='\r')

        if (m<=0): 
          acc=1
          for k in range (l+m+1,l-m):
            acc=acc*(k)
          prod, sign = (-1)**m/acc, -1
        else     : prod, sign = 1, 1
        
        myPml=Rml0[sign*m,:]*prod
        
        j=l**2+l+m

        if (m<=0): 
          acc=1
          for k in range (l+m+1,l-m):
            acc=acc*(k)
          nm = acc
        else     : 
          acc=1
          for k in range (l-m+1,l+m):
            acc=acc*(k)
          nm = 1/acc
        
        RY[:,j]=myPml.T * np.exp((1.j) * m * rphi) * np.sqrt(np.complex((2 * l + 1) / (4 * pi) * nm))

    #Check if everything is good
    self.check_NaN(RY,'RY')

    self.RY = RY
    self.rr = rr

  def compute_selected_points_reconstruction(self,c=np.array([None]),scale=1.):

    print(scale)

    if c.all() == None:
      if self.dim == 2: cx, cy     = self.cx * scale, self.cy*scale
      if self.dim == 3: cx, cy, cz = self.cx * scale, self.cy*scale, self.cz*scale
    else:
      if self.dim == 2: cx, cy     = c[0]*scale, c[1]*scale
      if self.dim == 3: cx, cy, cz = c[0]*scale, c[1]*scale, c[2]*scale

    self.x_Yc = np.real(self.b[0] + self.Y.dot(cx))
    self.y_Yc = np.real(self.b[1] + self.Y.dot(cy))
    if self.dim == 3: self.z_Yc = np.real(self.b[2] + self.Y.dot(cz))

    if scale > 1.:
      r = np.zeros((80,80))
    else:
      r = np.zeros(self.L)

    for i in range(self.n_points):
      if self.dim == 2: r[int(self.y_Yc[i])][int(self.x_Yc[i])] = 1
      if self.dim == 3: r[int(self.y_Yc[i])][int(self.x_Yc[i])][int(self.z_Yc[i])] = 1

    self.selected_points_reconstruction = r

  def compute_full_reconstruction(self,c=np.array([None]),scale=1.):

    if c.all() == None:
      if self.dim == 2: cx, cy     = self.cx * scale, self.cy*scale
      if self.dim == 3: cx, cy, cz = self.cx * scale, self.cy*scale, self.cz*scale
    else:
      if self.dim == 2: cx, cy     = c[0]*scale, c[1]*scale
      if self.dim == 3: cx, cy, cz = c[0]*scale, c[1]*scale, c[2]*scale

    self.x_RYc = (self.RY.dot(cx)).reshape(self.L)
    self.y_RYc = (self.RY.dot(cy)).reshape(self.L)
    if self.dim == 3: self.z_RYc = (self.RY.dot(cz)).reshape(self.L)

    if self.dim == 2: rs = np.sqrt(self.x_RYc**2 + self.y_RYc**2)
    if self.dim == 3: rs = np.sqrt(self.x_RYc**2 + self.y_RYc**2+self.z_RYc**2)
    
    self.form_reconstruction = (rs>self.rr)
    


