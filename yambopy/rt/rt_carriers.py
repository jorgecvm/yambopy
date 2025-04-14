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
       self.red_atomic_positions = lattice.red_atomic_positions
       self.atomic_numbers = lattice.atomic_numbers

       # Defining k-grid variables
       self.RT_kpoints = RT_kpt.T
       self.RT_kpt = RT_kpt

       # Defining carriers variables
       self.sel_band = sel_band
       self.RT_carriers_delta_f = RT_carriers_delta_f
       self.time = RT_carriers_delta_f.shape[0]
       self.kpt = RT_kpt.shape[1]
       self.nbands = int(RT_carriers_delta_f.shape[1]/self.kpt)
       self.carriers = np.zeros((self.time, self.kpt))

    def plot_carriers_ax(self, ax, time, vmin, vmax, cmap = 'PiYG', color_Wigner = 'black', lw_Wigner = 0.75, size = 50, limfactor=0.8, mode = 'raw', mode_BZ = 'repetition'):

        print("=== Plotting the carriers at a given time and band ===")

        # Reading carriers for a given band
        for t in range(self.time):
            for ik in range(self.kpt):
                self.carriers[t, ik] = self.RT_carriers_delta_f[t, ik*self.nbands + self.sel_band] 

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
           ax.scatter(x,y, c = self.carriers[time,:], vmin = -np.amax(self.carriers[time,:]), vmax = np.amax(self.carriers[time,:]), cmap = cmap, s = size, marker = 'H')

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

    def plot_carriers_on_band(self,ax,time,energies,path,cond_band,lpratio=5,f=None,verbose=1,size=5):

        eigs = energies.eigenvalues_ibz[0,:,24:28]

        symrel = [np.identity(3)]
        time_rev = False
 
        carriers_on_band = np.zeros([self.kpt,self.nbands])

        # Reading carriers for a given band
        for ib in range(self.nbands):
            for ik in range(self.kpt):
                carriers_on_band[ik, ib] = self.RT_carriers_delta_f[time, ik*self.nbands + ib] 

        # Additional input parameters for the SKW interpolator
        na = np.newaxis
        cell = (self.lat, self.red_atomic_positions, self.atomic_numbers)
        nelect = 0
        fermie = 0.0

        # Get dense kpoints along the path (controlled by path.intervals)
        kpoints_path =  path.get_klist()[:,:3]

        #interpolate carriers
        skw_carriers = SkwInterpolator(lpratio,self.kpoints_reduced,carriers_on_band[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        carriers_kpath = skw_carriers.interp_kpts(kpoints_path).eigens

        #interpolate energies
        skw_energies = SkwInterpolator(lpratio,self.kpoints_reduced,eigs[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        dft_energies = skw_energies.interp_kpts(kpoints_path).eigens

        path_car = get_path_car(red_car(path.kpoints,self.rlat),path)
        kpoints_path = path_car.get_klist()[:,:3]

        energy_bands = dft_energies[0].T
        self.distances = [0]
        distance = 0

        self.kpoints = np.array(kpoints_path)

        carriers_kpath = carriers_kpath[0]

        for nk in range(1,len(self.kpoints)):
            distance += np.linalg.norm(self.kpoints[nk]-self.kpoints[nk-1])
            self.distances.append(distance)
   
        for ib,band in enumerate(energy_bands):
            x = self.distances
            y = band-fermie
            ax.plot(x,y,c='black')

            dy = np.array(carriers_kpath[:,ib]*size)
 
            dy = np.abs(dy)
         
            if ib < cond_band:
               ax.fill_between(x,y+dy,y-dy,alpha=0.5,color='darkgreen')

            elif ib >= cond_band:
               ax.fill_between(x,y+dy,y-dy,alpha=0.5,color='magenta')

        exc_bands = YambopyBandStructure(dft_energies[0],kpoints_path,kpath=path_car)

        exc_bands.add_kpath_labels(ax)

        ax.set_ylabel('Energy (eV)')

    def plot_sum_carriers(self, ax, bx, regions, custom_cmap, size = 20, color_Wigner = 'black', lw_Wigner = 0.75, limfactor=0.8):

        # Reading carriers for a given band
        for t in range(self.time):
            for ik in range(self.kpt):
                self.carriers[t, ik] = self.RT_carriers_delta_f[t, ik*self.nbands + self.sel_band] 

        # Repeating the BZ
        kmesh_full, kmesh_idx = replicate_red_kmesh(self.kpoints_reduced,repx=range(-1,2),repy=range(-1,2))
        x,y = red_car(kmesh_full,self.rlat)[:,:2].T

        # Indexing the carriers to the expanded BZ
        self.carriers = self.carriers[:, kmesh_idx]

        # Defining the limits of the plot according to the reciprocal lattice parameters
        lim = np.max(self.rlat)*limfactor
        dlim = lim*1.1

        all_carriers = []  # To store carrier sums for each region

        weight_total = np.zeros(len(x))

        carriers_total = np.zeros((self.time, len(x)))

        for region_idx, (kx_center, ky_center, radius, region_label) in enumerate(regions, start = 1):
            for it in range(self.time):
               for ik in range(len(x)):
                   dx = x[ik] - kx_center
                   dy = y[ik] - ky_center
                   dist = np.sqrt(dx**2 + dy**2)

                   # Only assign if this point wasn't already assigned by a previous region
                   if dist < radius and weight_total[ik] == 0:
                       weight_total[ik] = region_idx
                       for it in range(self.time):
                           carriers_total[it, ik] = self.carriers[it, ik]

        ax.scatter(x, y, s=size, c=weight_total, cmap=custom_cmap, marker='H')
        ax.add_patch(BZ_Wigner_Seitz(self.lattice,color=color_Wigner,linewidth=lw_Wigner))

        time_step = np.linspace(0, 400, self.time)

        for region_idx, (kx_center, ky_center, radius, region_label) in enumerate(regions, start=1):
            mask = weight_total == region_idx  # Boolean array: True for points in this region

            # Extract and sum carriers for this region over time
            carriers_region = self.carriers[:, mask]  # shape: (time, num_kpoints_in_region)
            carriers_sum = np.sum(carriers_region, axis=1)  # sum over k-points → shape: (time,)

            region_color = custom_cmap(region_idx / (len(regions)))  # Normalize to [0,1] range

            bx.plot(time_step, carriers_sum, label=region_label, color=region_color)

        bx.legend(loc = 'upper right', frameon = False)

        lim = np.max(self.rlat)*limfactor
        dlim = lim*1.1

        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_xlabel('k$_{x}$ (a.u.)')
        ax.set_ylabel('k$_{y}$ (a.u.)')


        bx.set_xlim(np.min(time_step),np.max(time_step))
        bx.set_ylabel('Carriers (a.u.)')
        bx.set_xlabel('Time (fs)')

# To read data from netcdf files 
def read_netcdf(carriers_file):
    with nc.Dataset(carriers_file, 'r') as dataset:
         # Reading RT_kpt and RT_k_weight variables from ndb.RT_carriers
         RT_kpt = dataset.variables['RT_kpt'][:]
         RT_carriers_delta_f = dataset.variables['RT_carriers_delta_f'][:]
      
         return RT_kpt, RT_carriers_delta_f
