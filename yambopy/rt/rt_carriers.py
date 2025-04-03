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

       # Initialize the new array with dimensions (time, 144)
       self.Time = RT_carriers_delta_f.shape[0]
       self.Kpt = RT_kpt.shape[1]
       self.nbands = int(RT_carriers_delta_f.shape[1]/self.Kpt)

       Carriers = np.zeros((self.Time, self.Kpt))

       for t in range(self.Time):
           # Select the third value from every group of 4 along the RTstates dimension
           for ik in range(self.Kpt):
               Carriers[t, ik] = RT_carriers_delta_f[t, ik*self.nbands + sel_band]  # Selects the third value (index 2 of every group)

       print(RT_kpt.T)

       exit()

    def plot_carriers_ax(file_path):
 
        for t in range(RT_carriers_delta_f.shape[0]-1):
            # Select the third value from every group of 4 along the RTstates dimension
            for i in range(144):
                New_array[t, i] = RT_carriers_delta_f[t, i*X + selected_band]  # Selects the third value (index 2 of every group)
        print(New_array[399,:])


def read_netcdf(carriers_path):
    with nc.Dataset(carriers_path, 'r') as dataset:
         # Read RT_kpt and RT_k_weight
         RT_kpt = dataset.variables['RT_kpt'][:]
         RT_carriers_delta_f = dataset.variables['RT_carriers_delta_f'][:]
      
         return RT_kpt, RT_carriers_delta_f







