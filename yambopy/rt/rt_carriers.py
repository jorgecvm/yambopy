# 
# License-Identifier: GPL
#
# Copyright (C) 2025 The Yambo Team
#
# Authors: JC-V
#
# This file is part of the yambopy project
#
#
import netCDF4 as nc
from yambopy import *
from yambopy.plot  import *

class YamboRTcarriers():

    def __init__(self, carriers_file, save_path, sel_band):

       """
       Plot the RT carriers for a given time and band in the Brillouin zone (BZ).

       === Usage and variables ===

       >> Carriers = YamboRTcarriers(carriers_file,sel_band)
       >> Carriers.plot_carriers_ax(ax,carriers_file, Time, vmin, vmax, cmap)

       Input:
       :: carriers_file defines the ndb.RT_carriers file
       :: save_path defines the SAVE directory with file ns.db1
       :: sel_band defines the band that you want to plot (notice that if you have two valence bands and two conduction band, the first conduction band would be sel_band = 2)

       :: time defines time in femtoseconds 
       :: color_Wigner defines the color of the BZ Wigner Seitz cell
       :: lw_Wigner defines the linewidth of the BZ Wigner Seitz cell
       :: limfactor defines the size of the plot
       :: mode defines the type of the plot, there are two options: 'raw' (the data is plotted as computed) and 'interpolated' (the data is interpolated)
       :: mode_BZ defines how the BZ Wigner Seitz cell is plotted, there are two options: 'simple' (the BZ Wigner Seitz cell is plotted just at the center) and 'repetition' 
          (the BZ Wigner Seitz cells are plotted along the repeated BZ)
       
       Output:
       :: Plot of the carriers for a given time and band along the repeated BZ
       """

       print("=== Initializing the data ===")

       # Reading the kpoints and carriers as a function of time
       RT_kpt, RT_carriers_delta_f = read_netcdf(carriers_file)

       # Reading the k-points in reduced coordinates 
       lattice = YamboLatticeDB.from_db_file(filename=save_path)
       kpoints_reduced = lattice.red_kpoints

       # Defining lattice variables
       self.lattice = lattice
       self.kpoints_reduced = kpoints_reduced
       self.lat = lattice.lat
       self.rlat = lattice.rlat

       # Defining k-grid variables
       self.RT_kpoints = RT_kpt.T
       self.RT_kpt = RT_kpt

       # Defining carriers variables
       self.RT_carriers_delta_f = RT_carriers_delta_f
       self.time = RT_carriers_delta_f.shape[0]
       self.kpt = RT_kpt.shape[1]
       self.nbands = int(RT_carriers_delta_f.shape[1]/self.kpt)
       self.carriers = np.zeros((self.time, self.kpt))

       # Reading carriers for a given band
       for t in range(self.time):
           for ik in range(self.kpt):
               self.carriers[t, ik] = self.RT_carriers_delta_f[t, ik*self.nbands + sel_band] 

    def plot_carriers_ax(self, ax, time, vmin, vmax, cmap = 'PiYG', color_Wigner = 'black', lw_Wigner = 0.75, size = 50, limfactor=0.8, mode = 'raw', mode_BZ = 'repetition'):

        print("=== Plotting the carriers at a given time and band ===")

        # Repeating the BZ
        kmesh_full, kmesh_idx = replicate_red_kmesh(self.kpoints_reduced,repx=range(-1,2),repy=range(-1,2))
        x,y = red_car(kmesh_full,self.rlat)[:,:2].T

        # Indexing the carriers to the expanded BZ
        self.carriers = self.carriers[:, kmesh_idx]

        # Defining the limits of the plot according to the reciprocal lattice parameters
        lim = np.max(self.rlat)*limfactor
        dlim = lim*1.1

        # Mode = 'raw': Plotting data as computed
        if mode == 'raw':
           ax.scatter(x,y, c = self.carriers[time,:], vmin = vmin, vmax = vmax, cmap = cmap, s = size)

        # Mode = 'rbf': Plotting interpolated data
        elif mode == 'interpolated':

           # Defining interpolation variables
           from scipy.interpolate import Rbf
           npts = 100
           rbfi = Rbf(x,y,self.carriers[time,:],function='linear')
           x_int = y_int = np.linspace(-lim,lim,npts)
           carriers_int = np.zeros([npts,npts])

           # Plotting carriers in the interpolated expanded BZ
           for col in range(npts):
               carriers_int[:,col] = rbfi(x_int,np.ones_like(x_int)*y_int[col])
               s=ax.imshow(carriers_int.T,interpolation='bicubic',extent=[-lim,lim,-lim,lim], cmap = cmap)

        else: 
           raise ValueError("Mode has to be raw or interpolated")       
 

        # Mode = 'simple': The BZ Wigner Seitz cell appears just at the center
        if mode_BZ == 'simple':
           ax.add_patch(BZ_Wigner_Seitz(self.lattice,color=color_Wigner,linewidth=lw_Wigner))

        # Mode = 'simple': The BZ Wigner Seitz cell is repeated along the expanded BZ
        elif mode_BZ == 'repetition':

           # Defining repetition variables for the BZ Wigner Seitz cell 
           import itertools
           pairs = np.array(list(itertools.product([1, -1], repeat=2)))
           disp = np.array([self.lattice.rlat[0][0],self.lattice.rlat[0][0]*((self.lattice.rlat[1][1]*self.lattice.lat[0][0])/2.0)])

           # Plotting the BZ Wigner Seitz cell at the center
           ax.add_patch(BZ_Wigner_Seitz(self.lattice,color=color_Wigner,linewidth=lw_Wigner))
           disp = np.array([self.lattice.rlat[0][0],self.lattice.rlat[0][0]*((self.lattice.rlat[1][1]*self.lattice.lat[0][0])/2.0)])

           # Plotting the BZ Wigner Seitz cell along the expanded BZ
           for ip in range(len(pairs)):
               center_disp = pairs[ip]*disp
               ax.add_patch(BZ_Wigner_Seitz(self.lattice,center=(center_disp[0],center_disp[1]),color=color_Wigner,linewidth=lw_Wigner))

        else: 
           raise ValueError("Mode_BZ has to be simple or repetition")    

        # Plot variables
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_xlabel('k$_{x}$ (a.u.)')
        ax.set_ylabel('k$_{y}$ (a.u.)')

        print("=== Carriers plotted successfully ===")
           
# To read data from netcdf files 
def read_netcdf(carriers_file):
    with nc.Dataset(carriers_file, 'r') as dataset:
         # Reading RT_kpt and RT_k_weight variables from ndb.RT_carriers
         RT_kpt = dataset.variables['RT_kpt'][:]
         RT_carriers_delta_f = dataset.variables['RT_carriers_delta_f'][:]
      
         return RT_kpt, RT_carriers_delta_f
