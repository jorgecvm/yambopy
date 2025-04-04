# Copyright (c) 2018, Alejandro Molina-Sanchez
# All rights reserved.
#
# This file is part of the yambopy project
#
import netCDF4 as nc
from yambopy import *
from yambopy.plot  import *
import os

class YamboRTcarriers():
    """

    """
    def __init__(self, carriers_path, sel_band):
       
       print("=== Initializing the data ===")

       RT_kpt, RT_carriers_delta_f = read_netcdf(carriers_path)

       self.RT_kpoints = RT_kpt.T

       # Initialize the new array with dimensions (time, 144)
       self.RT_kpt = RT_kpt
       self.Time = RT_carriers_delta_f.shape[0]
       self.Kpt = RT_kpt.shape[1]
       self.nbands = int(RT_carriers_delta_f.shape[1]/self.Kpt)

       self.Carriers = np.zeros((self.Time, self.Kpt))

       for t in range(self.Time):
           # Select the third value from every group of 4 along the RTstates dimension
           for ik in range(self.Kpt):
               self.Carriers[t, ik] = RT_carriers_delta_f[t, ik*self.nbands + sel_band] 


    def plot_carriers_ax(self,ax,file_path):

        prueba = YamboLatticeDB.from_db_file(filename='SAVE/ns.db1')

        Kpints_bien = prueba.red_kpoints
        self.rlat = prueba.rlat

        kmesh_full, kmesh_idx = replicate_red_kmesh(Kpints_bien,repx=range(-1,2),repy=range(-1,2))
        x,y = red_car(kmesh_full,self.rlat)[:,:2].T

        self.Carriers[:,:] = self.Carriers[:,kmesh_idx]
       
        ax.scatter(x,y)
  

        '''
        for ik in range(self.Kpt): 
            ax.scatter(Kpints_bien[ik,0],Kpints_bien[ik,1], c = self.Carriers[399,ik], vmin = -2.0e-4, vmax = 2.0e-4, cmap = 'viridis', s = 30)
            print(self.Carriers[399,ik])
        '''

def read_netcdf(carriers_path):
    with nc.Dataset(carriers_path, 'r') as dataset:
         # Read RT_kpt and RT_k_weight
         RT_kpt = dataset.variables['RT_kpt'][:]
         RT_carriers_delta_f = dataset.variables['RT_carriers_delta_f'][:]
      
         return RT_kpt, RT_carriers_delta_f







