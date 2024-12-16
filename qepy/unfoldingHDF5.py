# Copyright (C) 2024 Henrique Pereira Coutada Miranda, Alejandro Molina-Sanchez, Jorge Cervantes-Villanueva, Fulvio Paleari
# All rights reserved.
#
# This file is part of yambopy
#
# Unfolding of supercell band structure to a effective band structure (EBS) projecting into a primitive cell.
# Program adapted for reading Quantum Espresso output files.
#
# Authors: Alejandro Molina-Sanchez, Henrique Miranda, Jorge Cervantes-Villanueva and Fulvio Paleari
#
# Future developments will be:
# 
# - Assignment of the correspondence G-vectors to g-vectors without reading the PC g-vectors
#

from qepy import *
import numpy as np
from sys import stdout
import h5py

class UnfoldingHDF5():

    def __init__(self,prefix_pc,prefix_sc,path_pc='.',path_sc='.',spin="none",band_min=0,sc_rotated=False,compute_projections=True):
        """ 
        Initialize the structure with data from the unit cell and the super cell

        spin = "none" or "noncol" for nspin = 1 (noSOC) and nspin = 4 (SOC) calculations
        band_min = If band_min = 0, it computes the projection of all the bands. If band_min != 0, it computes the projection of the bands nbands-band_min
        sc_rotated = If sc_rotated = False, it takes the matrix identity. If sc_rotated = 3x3 matrix, it takes the rotation matrix defined by the user.
        compute_projections: If compute_projections = True, it compute the projections and save them in a .npy file. If compute_projections = False, it loads the previous .npy file. 
        """

        # Prefix and path of unit cell and supercell
        self.prefix_pc = prefix_pc  
        self.prefix_sc = prefix_sc
        self.path_pc   = path_pc 
        self.path_sc   = path_sc
       
        # Reading unit cell and supercell database from QE
        pc_xml = PwXML(prefix=self.prefix_pc,path=self.path_pc) 
        sc_xml = PwXML(prefix=self.prefix_sc,path=self.path_sc)

        # Number of kpoints of the unit cell and supercell
        self.nkpoints_pc = pc_xml.nkpoints 
        self.nkpoints_sc = sc_xml.nkpoints 

        # List of kpoints of the supercell 
        self.kpoints = sc_xml.kpoints  

        # Number of bands of the unit cell and supercell
        self.nbands_pc = pc_xml.nbands  
        self.nbands_sc = sc_xml.nbands 

        # This indicates the code to compute the projections of nbands - band_min bands
        self.band_min  = band_min 

        # Condition to control the number of computed bands  
        if self.band_min > self.nbands_sc:
           raise Exception("Minimum of bands larger than total number of bands")

        # Reciprocal lattice of the unit cell and supercell in cartesian coordiantes
        self.rcell_pc = array(pc_xml.rcell)/pc_xml.cell[0][0] 
        self.rcell_sc = array(sc_xml.rcell)/sc_xml.cell[0][0]
   
        # Eigenvalues of the unit cell and supercell
        self.eigen_pc = array(pc_xml.eigen1) 
        self.eigen_sc = array(sc_xml.eigen1) 

        # Format to save Miller indices
        format_string = "%12.4lf %12.4lf %12.4lf" 
        n_decs = 8 

        # Distance of the k-path
        kpoints_dists = calculate_distances(self.kpoints) 
       
        # Condition to use a rotation matrix if needed 
        if sc_rotated is False:
            self.rot = np.array([ [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0] ])

        elif isinstance(sc_rotated, np.ndarray) and sc_rotated.shape == (3,3):
            self.rot = sc_rotated

        else:
            raise Exception("Input for sc_rotated must be False or a 3x3 matrix")

        # Array to save the projections
        self.projection = zeros([self.nkpoints_sc,self.nbands_sc-self.band_min]) 


        " Calculation of the projections "

        # Condition to compute the projections or to use the saved ones  
        if compute_projections == True:

           print("Dictionary of G-vectors and projection")

           # Loop along the kpoints of the supercell  
           for ik in range(self.nkpoints_sc):
               load(ik,self.nkpoints_sc)

               # Loading the data of each kpoint of the unit cell and supercell 
               f_pc = h5py.File('%s/%s.save/wfc%01d.hdf5' % (self.path_pc,self.prefix_pc,(ik + 1)), 'r') 
               f_sc = h5py.File('%s/%s.save/wfc%01d.hdf5' % (self.path_sc,self.prefix_sc,(ik + 1)), 'r') 

               # Dimension of the Miller indices
               self.ng_pc = int(f_pc.attrs['igwx'])
               self.ng_sc = int(f_sc.attrs['igwx']) 

               # Creating a dictionary to collect key-value pairs
               g_sc = dict()
               g_sc_int = dict()
           
               # Loading Miller indices
               mill_sc = f_sc['MillerIndices']
               mill_pc = f_pc['MillerIndices']
        
               # Loop along the number of Miller indices of the super cell
               for ig in arange(self.ng_sc):
              
                   # Definition of Miller indices in order to reconstruct the k-points (to rotate it if neccessary) 
                   # and to save them in the dictionary associated to an integer
                   h,k,l = mill_sc[ig,0], mill_sc[ig,1], mill_sc[ig,2]
                   g_sc_int[(int(h),int(k),int(l))] = ig
                   w = h*self.rcell_sc[:][0] + k*self.rcell_sc[:][1] + l*self.rcell_sc[:][2]
                   w = dot(self.rot,w)
                   w = np.around(w, decimals = n_decs) + array([0,0,0])
                   w = format_string % (abs(w[0]), abs(w[1]), abs(w[2]))
                   g_sc[w] = ig
                   
               # To check in the following loop if the k-point of the super cell is found in the unit cell    
               g_contain = [0]*self.ng_pc

               # Loop along the number of Miller indices of the unit cell
               for ig in arange(self.ng_pc):

                   # Definition of Miller indices in order to reconstruct the k-points
                   h,k,l = mill_pc[ig,0], mill_pc[ig,1], mill_pc[ig,2]
                   w = h*self.rcell_pc[:][0] + k*self.rcell_pc[:][1] + l*self.rcell_pc[:][2]
                   w = np.around(w, decimals = n_decs) + array([0,0,0])
                   w = format_string % (abs(w[0]), abs(w[1]), abs(w[2]))

                   # Checking if the k-point in the supercell is found in the unit cell,
                   # if missing, the projection will be wrong.
                   try:
                       g_contain[ig] = g_sc[w]
                   except KeyError:
                       print("Missing k-point %d" % ig)
           

               # Condition to read the eigenvectors for nspin = 1 or nspin = 4 in QE (To implement spin == col, i.e., nspin = 2 in QE)
               if spin == "none" or spin == "noncol":
              
                  # Loading eigenvectors of the super cell
                  evc_sc = f_sc['evc']
                  eivecs = []

                  # Loop along the number of bands indicated by the user to be computed
                  for ib in range(self.band_min,self.nbands_sc):
                      eivec = evc_sc[ib, :]
                      
                      # Rewriting the eigenvectors to manage them properly
                      eivec_complex = [complex(eivec[i], eivec[i+1]) for i in range(0, len(eivec), 2)]

                      eivecs.append(eivec_complex)
                        
                      # Defining specifically ib == 0 since it presents one component instead of two as the rest of values
                      if ib==0:
                         x = 0.0
                         for ig in range(self.ng_sc):
                             x += eivecs[-1][ig]*eivecs[-1][ig].conjugate()
 
               # Condition to compute the projections for nspin = 1 or spin = 4 in QE (To implement spin == col, i.e., nspin = 2 in QE)
               if spin == "none" or spin == "noncol":

                  # Loop along the number of bands indicated by the user to be computed
                  for ib in range(self.nbands_sc-self.band_min): 
                      x = 0.0
                      for ig in range(self.ng_pc): 

                          # Computing the projection between the unit cell and the super cell
                          x += eivecs[ib][g_contain[ig]]*(eivecs[ib][g_contain[ig]].conjugate())

                      # If the value is less than a threshold, the projection is set to zero (to avoid ficticious points when plotting)
                      if abs(x) < 1e-2:

                         self.projection[ik,ib] = 0.0
                     
                      else:

                         self.projection[ik,ib] = abs(x)
               
               # Saving the data to avoid recomputing the projections twice
               np.save('projections',self.projection)


        # Loading the projections when already computed
        if compute_projections == False:

           self.projection = np.load('projections.npy')

        return print("Projections calculated successfully!")

    def plot_eigen_ax(self,ax,path=[],xlim=(),ylim=()):
        """
        Provisional plot function for quick visualization of the data. Useful for small calculations. 
        For large calculations is more useful loading the .npy and plot it with another script.
        """

        # Getting the data of the defined path
        if path:
            if isinstance(path,Path):
                path = path.get_indexes()
            ax.set_xticks( *list(zip(*path)) )
        ax.set_ylabel('E (eV)')

        # Computing the distance of the path 
        kpoints_dists = calculate_distances(self.kpoints)

        # Defining labels
        ticks, labels = list(zip(*path))
        ax.set_xticks([kpoints_dists[t] for t in ticks])
        ax.set_xticklabels(labels)
        ax.set_ylabel('E (eV)')

        # Plotting high-symmetry vertical lines
        for t in ticks:
            ax.axvline(kpoints_dists[t],c='k',lw=2)
        ax.axhline(0,c='k',lw=1)

        # Plotting the band for the unit cell
        for ib in range(self.nbands_pc):
           ax.plot(kpoints_dists,self.eigen_pc[:,ib],'k--',lw=0.5)

        # Plotting the projection of the super cell 
        for ib in range(self.nbands_sc-self.band_min):
           #ax.plot(kpoints_dists,self.eigen_sc[:,ib],'darkgreen',lw=0.2)
           ax.scatter(kpoints_dists,self.eigen_sc[:,ib+self.band_min],s=self.projection[:,ib]*5,color='navy',edgecolor = None)

        # Establishing x and y axis limits 
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

# Load bar
def load(x,n):
    bar_length = 100
    x+=1
    ratio = x/float(n)
    c = int(ratio * bar_length)
    stdout.write("["+"="*c+" "*(bar_length-c)+"] %03.3f%%" % (ratio*100))
    if (x==n): stdout.write("\n")
    stdout.flush()
    stdout.write("\r")
