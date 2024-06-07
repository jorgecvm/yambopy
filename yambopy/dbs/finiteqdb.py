from yambopy.units import *
from yambopy.plot.plotting import add_fig_kwargs,BZ_Wigner_Seitz
from yambopy.plot.bandstructure import *
from yambopy.lattice import replicate_red_kmesh, calculate_distances, get_path, car_red
from yambopy.tools.funcs import gaussian, lorentzian
from yambopy.dbs.savedb import *
from yambopy.dbs.latticedb import *
from yambopy.dbs.electronsdb import *
from yambopy.dbs.qpdb import *
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
import time



class YamboExcitonFiniteQ(YamboSaveDB):
    """ Read the excitonic states database from yambo

        Exciton eigenvectors are arranged as eigenvectors[i_exc, i_kvc]
        Transitions are unpacked in table[ i_k, i_v, i_c, i_s_c, i_s_v ] (last two are spin indices)
    """
    def __init__(self,lattice,Qpt,eigenvalues,l_residual,r_residual,spin_pol='no',car_qpoint=None,q_cutoff=None,table=None,eigenvectors=None):
        if not isinstance(lattice,YamboLatticeDB):
            raise ValueError('Invalid type for lattice argument. It must be YamboLatticeDB')

        self.Qpt = int(Qpt)
        self.lattice = lattice
        self.eigenvalues = eigenvalues
        self.l_residual = l_residual
        self.r_residual = r_residual
        #optional
        self.car_qpoint = car_qpoint
        self.q_cutoff = q_cutoff
        self.table = table
        self.eigenvectors = eigenvectors
        self.spin_pol = spin_pol
        self.nqpoints = lattice.ibz_nkpoints

    @classmethod
    def from_db_file(cls,lattice,filename='ndb.BS_diago_Q1',folder='.'):
        """ initialize this class from the set of Q# files
        """
        path_filename = os.path.join(folder,filename)
        if not os.path.isfile(path_filename):
            raise FileNotFoundError("File %s not found in YamboExcitonDB"%path_filename)
       #print('This class reads the Q-files of BSE and store them in a list')

        # Qpoint
        Qpt = filename.split("Q",1)[1]
        nqpoints = lattice.ibz_nkpoints
        # All data inside this list
        Q_container = []
        
        for iq in range(nqpoints):
            filename_q = filename[:-1] + '%s' % (iq+1)
            path_filename = os.path.join(folder,filename_q)
            Qpt = filename_q.split("Q",1)[1]
            with Dataset(path_filename) as database:

                 if 'BS_left_Residuals' in list(database.variables.keys()):
                     #residuals
                    rel,iml = database.variables['BS_left_Residuals'][:].T
                    rer,imr = database.variables['BS_right_Residuals'][:].T
                    l_residual = rel+iml*I
                    r_residual = rer+imr*I
                 if 'BS_Residuals' in list(database.variables.keys()):
                     #residuals
                     rel,iml,rer,imr = database.variables['BS_Residuals'][:].T
                     l_residual = rel+iml*I
                     r_residual = rer+imr*I

                 # Finite momentum
                 car_qpoint = lattice.ibz_kpoints[iq]/lattice.alat
                 # database.variables['Q-point'][:]/lattice.alat check if
                 # correct the car_qpoint

                 #energies
                 eig =  database.variables['BS_Energies'][:]*ha2ev
                 eigenvalues = eig[:,0]+eig[:,1]*I
                     
                 #eigenvectors
                 table = None
                 eigenvectors = None
                 if 'BS_EIGENSTATES' in database.variables:
                     eiv = database.variables['BS_EIGENSTATES'][:]
                     eiv = eiv[:,:,0] + eiv[:,:,1]*I
                     eigenvectors = eiv
                     table = database.variables['BS_TABLE'][:].T.astype(int)
                 
                 table = table
                 eigenvectors = eigenvectors
                 spin_vars = [int(database.variables['SPIN_VARS'][:][0]), int(database.variables['SPIN_VARS'][:][1])]
                 if spin_vars[0] == 2 and spin_vars[1] == 1:
                    spin_pol = 'pol'
                 else:
                    spin_pol = 'no'
                 # Check if Coulomb cutoff is present
                 path_cutoff = os.path.join(path_filename.split('ndb',1)[0],'ndb.cutoff')  
                 q_cutoff = None
                 if os.path.isfile(path_cutoff):
                    with Dataset(path_cutoff) as database:
                         bare_qpg = database.variables['CUT_BARE_QPG'][:]
                         bare_qpg = bare_qpg[:,:,0]+bare_qpg[:,:,1]*I
                         q_cutoff = np.abs(bare_qpg[0,int(Qpt)-1])
        
                 qdata = cls(lattice,Qpt,eigenvalues,l_residual,r_residual,spin_pol,q_cutoff=q_cutoff,car_qpoint=car_qpoint,table=table,eigenvectors=eigenvectors)
                 Q_container.append(qdata)
        
        return Q_container

    @property
    def unique_vbands(self):
        return np.unique(self.table[:,1]-1)

    @property
    def unique_cbands(self):
        return np.unique(self.table[:,2]-1)

    @property
    def transitions_v_to_c(self):
        """Compute transitions from valence to conduction"""
        if hasattr(self,"_transitions_v_to_c"): return self._transitions_v_to_c
        uniq_v = self.unique_vbands
        uniq_c = self.unique_cbands
        transitions_v_to_c = dict([ ((v,c),[]) for v,c in product(uniq_v,uniq_c) ])

        #add elements to dictionary
        kidx = set()
        for eh,kvc in enumerate(self.table-1):
            k,v,c = kvc
            kidx.add(k)
            transitions_v_to_c[(v,c)].append((k,eh))
        self.nkpoints = len(kidx)

        #make an array 
        for t,v in list(transitions_v_to_c.items()):
            if len(np.array(v)):
                transitions_v_to_c[t] = np.array(v)
            else:
                del transitions_v_to_c[t]

        self._transitions_v_to_c = transitions_v_to_c 
        return transitions_v_to_c

    @property
    def nkpoints(self): return max(self.table[:,0])

    @property
    def nvbands(self): return len(self.unique_vbands)

    @property
    def ncbands(self): return len(self.unique_cbands)

    @property
    def nbands(self): return self.ncbands+self.nvbands

    @property
    def mband(self): return max(self.unique_cbands)+1
 
    @property
    def ntransitions(self): return len(self.table)

    @property
    def nexcitons(self): return len(self.eigenvalues)
    
    @property
    def start_band(self): return min(self.unique_vbands)

    def write_sorted(self,prefix='yambo'):
        """
        Write the sorted energies and intensities to a file
        """
        #get intensities
        eig = self.eigenvalues.real
        intensities = self.get_intensities()

        #get sorted energies
        sort_e, sort_i = self.get_sorted()     

        #write excitons sorted by energy
        with open('%s_E.dat'%prefix, 'w') as f:
            for e,n in sort_e:
                f.write("%3d %12.8lf %12.8e\n"%(n+1,e,intensities[n])) 

        #write excitons sorted by intensities
        with open('%s_I.dat'%prefix,'w') as f:
            for i,n in sort_i:
                f.write("%3d %12.8lf %12.8e\n"%(n+1,eig[n],i)) 

    def get_nondegenerate(self,eps=1e-4):
        """
        get a list of non-degenerate excitons
        """
        non_deg_e   = [0]
        non_deg_idx = [] 

        #iterate over the energies
        for n,e in enumerate(self.eigenvalues):
            if not np.isclose(e,non_deg_e[-1],atol=eps):
                non_deg_e.append(e)
                non_deg_idx.append(n)

        return np.array(non_deg_e[1:]), np.array(non_deg_idx)

    def get_intensities(self):
        """
        get the intensities of the excitons
        """
        intensities = self.l_residual*self.r_residual
        intensities /= np.max(intensities)
        return intensities

    def get_sorted(self):
        """
        Return the excitonic weights sorted according to energy and intensity
        """
        #get intensities
        eig = self.eigenvalues.real
        intensities = self.get_intensities()

        #list ordered with energy
        sort_e = sorted(zip(eig, list(range(self.nexcitons))))

        #list ordered with intensity
        sort_i = sorted(zip(intensities, list(range(self.nexcitons))),reverse=True)

        return sort_e, sort_i 

    def get_degenerate(self,index,eps=1e-4):
        """
        Get degenerate excitons
        
        Args:
            eps: maximum energy difference to consider the two excitons degenerate in eV
        """
        energy = self.eigenvalues[index-1]
        excitons = [] 
        for n,e in enumerate(self.eigenvalues):
            if np.isclose(energy,e,atol=eps):
                excitons.append(n+1)
        return excitons

    def exciton_bs(self,energies,path,excitons=(0,),debug=False):
        """
        Calculate exciton band-structure
            
            Arguments:
            energies -> can be an instance of YamboSaveDB or YamboQBDB
            path     -> path in reduced coordinates in which to plot the band structure
            exciton  -> exciton index to plot
            spin     -> ??
        """
        if self.eigenvectors is None:
            raise ValueError('This database does not contain Excitonic states,'
                              'please re-run the yambo BSE calculation with the WRbsWF option in the input file.')
        if isinstance(excitons, int):
            excitons = (excitons,)
        #get full kmesh
        kpoints = self.lattice.red_kpoints
        path = np.array(path)

        rep = list(range(-1,2))
        kpoints_rep, kpoints_idx_rep = replicate_red_kmesh(kpoints,repx=rep,repy=rep,repz=rep)
        band_indexes = get_path(kpoints_rep,path,debug=debug)
        band_kpoints = kpoints_rep[band_indexes] 
        band_indexes = kpoints_idx_rep[band_indexes]

        if debug:
            import matplotlib.pyplot as plt
            for i,k in zip(band_indexes,band_kpoints):
                x,y,z = k
                plt.text(x,y,i) 
            plt.scatter(kpoints_rep[:,0],kpoints_rep[:,1])
            plt.plot(path[:,0],path[:,1],c='r')
            plt.scatter(band_kpoints[:,0],band_kpoints[:,1])
            plt.show()
            exit()

        #get eigenvalues along the path
        if isinstance(energies,(YamboSaveDB,YamboElectronsDB)):
            #expand eigenvalues to the full brillouin zone
            # SPIN-UP CHANNEL ONLY. Check with BSE WFs
            energies = energies.eigenvalues[0,self.lattice.kpoints_indexes]
            
        elif isinstance(energies,YamboQPDB):
            #expand the quasiparticle energies to the bull brillouin zone
            pad_energies = energies.eigenvalues_qp[self.lattice.kpoints_indexes]
            min_band = energies.min_band
            nkpoints, nbands = pad_energies.shape
            energies = np.zeros([nkpoints,energies.max_band])
            energies[:,min_band-1:] = pad_energies 
        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))

        weights = self.get_exciton_weights(excitons)      
        energies = energies[band_indexes]
        weights  = weights[band_indexes]

        #make top valence band to be zero
        energies -= max(energies[:,max(self.unique_vbands)])
        
        return np.array(band_kpoints), energies, weights 

    def get_exciton_weights(self,excitons):
        """get weight of state in each band"""
        weights = np.zeros([self.nkpoints,self.mband])
        for exciton in excitons:
            #get the eigenstate
            eivec = self.eigenvectors[exciton-1]
            #add weights
            sum_weights = 0
            for t,kcv in enumerate(self.table):
                k,v,c = kcv[0:3]-1    # This is bug's source between yambo 4.4 and 5.0 
                this_weight = abs2(eivec[t])
                weights[k,c] += this_weight
                weights[k,v] += this_weight
                sum_weights += this_weight
            if abs(sum_weights - 1) > 1e-3: raise ValueError('Excitonic weights does not sum to 1 but to %lf.'%sum_weights)
 
        return weights

    def get_exciton_weights_finiteq(self,excitons,iq,kindx):

        ''' This function calculate the weights of each band for given k and
        k-q. Still not sure if it is correct
        '''

        weights_v = np.zeros([self.nkpoints,self.nbands]) ### Why not to use # of bands instead of all the bands?
        weights_c = np.zeros([self.nkpoints,self.nbands])

        for exciton in excitons:
            #get the eigenstate
            eivec = self.eigenvectors[exciton-1]

            #add weights
            sum_weights = 0
            for t,kcv in enumerate(self.table):
                k = kcv[0] - 1
                v = kcv[1] - self.unique_vbands[0] - 1
                c = kcv[2] - self.unique_vbands[0] - 1
                #k,v,c   = kcv[0:3]-1    # This is bug's source between yambo 4.4 and 5.0
                k_v,k_c = k, kindx.qindx_X[iq-1,k,0] - 1
                this_weight = abs2(eivec[t])
                weights_v[k_v,v] += this_weight
                weights_c[k_c,c] += this_weight
                if weights_c[k_c,c] > 0.02:
                   print(k_v, self.lattice.red_kpoints[k_v], k_c, self.lattice.red_kpoints[k_c], v, c, weights_c[k_c,c]) 

                sum_weights += this_weight 
            if abs(sum_weights - 1) > 1e-3: raise ValueError('Excitonic weights does not sum to 1 but to %lf.'%sum_weights)

        return weights_v, weights_c

    
    def get_exciton_total_weights(self,excitons):
        """get weight of state in each band"""
        total_weights = np.zeros(self.nkpoints)
        for exciton in excitons:
            #get the eigenstate
            eivec = self.eigenvectors[exciton-1]
            #add weights
            sum_weights = 0
            for t,kcv in enumerate(self.table):
                k,c,v = kcv[0:3]
                total_weights[k-1] += abs2(eivec[t])
            if abs(sum(total_weights) - 1) > 1e-3: raise ValueError('Excitonic weights does not sum to 1 but to %lf.'%sum_weights)
 
        return total_weights

    def get_exciton_transitions(self,excitons):
        """get weight of state in each band"""
        # Double check the part of the array w_k_v_to_c
        # We should comment more this part
        #weights = np.zeros([self.nkpoints,self.mband])
        w_k_v_to_c = np.zeros([self.nkpoints,self.nvbands,self.ncbands])
        v_min = self.unique_vbands[0]
        c_min = self.unique_cbands[0]
        for exciton in excitons:
            #get the eigenstate
            eivec = self.eigenvectors[exciton-1]
            #add weights
            #sum_weights = 0
            for t,kcv in enumerate(self.table):
                k,c,v = kcv-1
                #k,v,c = kcv-1                                 # bug?? Double-check
                this_weight = abs2(eivec[t])
                w_k_v_to_c[k,v-v_min,c-c_min] = this_weight   # new
            #if abs(sum_weights - 1) > 1e-3: raise ValueError('Excitonic weights does not sum to 1 but to %lf.'%sum_weights)
 
        #return weights, w_k_v_to_c
        return w_k_v_to_c

    def get_exciton_2D(self,excitons,f=None):
        """get data of the exciton in 2D"""
        weights = self.get_exciton_weights(excitons)
        #sum all the bands
        weights_bz_sum = np.sum(weights,axis=1)
        if f: weights_bz_sum = f(weights_bz_sum)

        kmesh_full, kmesh_idx = replicate_red_kmesh(self.lattice.red_kpoints,repx=range(-1,2),repy=range(-1,2))
        x,y = red_car(kmesh_full,self.lattice.rlat)[:,:2].T
        weights_bz_sum = weights_bz_sum[kmesh_idx]
        return x,y,weights_bz_sum

    def get_exciton_2D_spin_pol(self,excitons,f=None):
        """get data of the exciton in 2D for spin polarized calculations"""
        weights_up, weights_dw = self.get_exciton_weights_spin_pol(excitons)

        #sum all the bands
        weights_bz_sum_up = np.sum(weights_up,axis=1)
        weights_bz_sum_dw = np.sum(weights_dw,axis=1)

        if f: weights_bz_sum_up = f(weights_bz_sum_up)
        if f: weights_bz_sum_dw = f(weights_bz_sum_dw)

        kmesh_full, kmesh_idx = replicate_red_kmesh(self.lattice.red_kpoints,repx=range(-1,2),repy=range(-1,2))
        x,y = red_car(kmesh_full,self.lattice.rlat)[:,:2].T
        weights_bz_sum_up = weights_bz_sum_up[kmesh_idx]
        weights_bz_sum_dw = weights_bz_sum_dw[kmesh_idx]

        return x,y,weights_bz_sum_up,weights_bz_sum_dw
 
    def get_exciton_2D_finite_q(self,excitons,iq,kindx,f=None):
        """get data of the exciton in 2D for finite-q"""
        print(self.Qpt)
        weights_v, weights_c  = self.get_exciton_weights_finiteq(excitons,self.Qpt,kindx) 

        print(weights_c)
        #sum all the bands for valence and conduction
        weights_v_bz_sum = np.sum(weights_v,axis=1)
        weights_c_bz_sum = np.sum(weights_c,axis=1)

        if f: weights_v_bz_sum = f(weights_v_bz_sum)
        if f: weights_c_bz_sum = f(weights_c_bz_sum)

        print(weights_c_bz_sum)


        kmesh_full, kmesh_idx = replicate_red_kmesh(self.lattice.red_kpoints,repx=range(0,1),repy=range(0,1))
        x,y = red_car(kmesh_full,self.lattice.rlat)[:,:2].T
        weights_v_bz_sum = weights_v_bz_sum[kmesh_idx]
        weights_c_bz_sum = weights_c_bz_sum[kmesh_idx]

        return x,y,weights_v_bz_sum,weights_c_bz_sum

    def plot_exciton_2D_ax(self,ax,excitons,f=None,mode='hexagon',limfactor=0.8,spin_pol=None,**kwargs):
        """
        Plot the exciton weights in a 2D Brillouin zone
       
           Arguments:
            excitons -> list of exciton indexes to plot
            f -> function to apply to the exciton weights. Ex. f=log will compute the 
                 log of th weight to enhance the small contributions
            mode -> possible values are 'hexagon'/'square' to use hexagons/squares as markers for the 
                    weights plot and 'rbf' to interpolate the weights using radial basis functions.
            limfactor -> factor of the lattice parameter to choose the limits of the plot 
            scale -> size of the markers
        """
        if spin_pol == None: print('Plotting exciton mad in 2D axis for spin polarization: %s' % spin_pol)

        if spin_pol == None:
           x,y,weights_bz_sum_up,weights_bz_sum_dw = self.get_exciton_2D_spin_pol(excitons,f=f)
        else:
           x,y,weights_bz_sum = self.get_exciton_2D(excitons,f=f)

        weights_bz_sum=weights_bz_sum/np.max(weights_bz_sum)
        
        #filter points outside of area
        lim = np.max(self.lattice.rlat)*limfactor
        dlim = lim*1.1
        if spin_pol == None:
           filtered_weights_up = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_bz_sum_up) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
           filtered_weights_dw = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_bz_sum_dw) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
           x,y,weights_bz_sum_up = np.array(filtered_weights_up).T
           x,y,weights_bz_sum_dw = np.array(filtered_weights_dw).T
        else:
           filtered_weights = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_bz_sum) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
           x,y,weights_bz_sum = np.array(filtered_weights).T
        # Add contours of BZ
        ax.add_patch(BZ_Wigner_Seitz(self.lattice))

        #plotting
        if mode == 'hexagon': 
            scale = kwargs.pop('scale',1)
            if spin_pol == 'up':
               s=ax.scatter(x,y,s=scale,marker='H',c=weights_bz_sum_up,rasterized=True,**kwargs)
            elif spin_pol == 'dw':
               s=ax.scatter(x,y,s=scale,marker='H',c=weights_bz_sum_dw,rasterized=True,**kwargs)
            else:
               s=ax.scatter(x,y,s=scale,marker='H',c=weights_bz_sum,rasterized=True,**kwargs)
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
        elif mode == 'square': 
            scale = kwargs.pop('scale',1)
            if spin_pol == 'up':
               s=ax.scatter(x,y,s=scale,marker='s',c=weights_bz_sum_up,rasterized=True,**kwargs)
            elif spin_pol == 'dw':
               s=ax.scatter(x,y,s=scale,marker='s',c=weights_bz_sum_dw,rasterized=True,**kwargs)
            else:
               s=ax.scatter(x,y,s=scale,marker='s',c=weights_bz_sum,rasterized=True,**kwargs)
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
        elif mode == 'rbf':
            from scipy.interpolate import Rbf
            npts = kwargs.pop('npts',100)
            interp_method = kwargs.pop('interp_method','bicubic')
            if spin_pol == 'up':
               rbfi = Rbf(x,y,weights_bz_sum_up,function='linear')
               x = y = np.linspace(-lim,lim,npts)
               weights_bz_sum_up = np.zeros([npts,npts])
            elif spin_pol == 'dw':
               rbfi = Rbf(x,y,weights_bz_sum_dw,function='linear')
               x = y = np.linspace(-lim,lim,npts)
               weights_bz_sum_dw = np.zeros([npts,npts])
            else:
               rbfi = Rbf(x,y,weights_bz_sum,function='linear')
               x = y = np.linspace(-lim,lim,npts)
               weights_bz_sum = np.zeros([npts,npts])

            for col in range(npts):
                if spin_pol == 'up':
                   weights_bz_sum_up[:,col] = rbfi(x,np.ones_like(x)*y[col])
                elif spin_pol == 'dw':
                   weights_bz_sum_dw[:,col] = rbfi(x,np.ones_like(x)*y[col])
                else:
                   weights_bz_sum[:,col] = rbfi(x,np.ones_like(x)*y[col])
            # NB we have to take the transpose of the imshow data to get the correct plot
            if spin_pol == 'up':
               s=ax.imshow(weights_bz_sum_up.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim])
            elif spin_pol == 'dw':
               s=ax.imshow(weights_bz_sum_dw.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim])
            else:
               s=ax.imshow(weights_bz_sum.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim])
        title = kwargs.pop('title',str(excitons))

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
       
        return ax,s

 
    def plot_exciton_2D_ax_finiteq(self,ax,bx,cx,iq,kindx,excitons,f=None,mode='hexagon',limfactor=0.8,spin_pol=None,**kwargs):
        """
        Plot the exciton weights in a 2D Brillouin zone for finite-q where the plots in valence and conduction are different
       

           Arguments:
            excitons -> list of exciton indexes to plot
            f -> function to apply to the exciton weights. Ex. f=log will compute the 
                 log of th weight to enhance the small contributions
            mode -> possible values are 'hexagon'/'square' to use hexagons/squares as markers for the 
                    weights plot and 'rbf' to interpolate the weights using radial basis functions.

            limfactor -> factor of the lattice parameter to choose the limits of the plot 
            scale -> size of the markers
        """
        
        x,y,weights_v_bz_sum,weights_c_bz_sum = self.get_exciton_2D_finite_q(excitons,self.Qpt,kindx,f=f)

        weights_v_bz_sum=weights_v_bz_sum/np.max(weights_v_bz_sum)
        weights_c_bz_sum=weights_c_bz_sum/np.max(weights_c_bz_sum)

        weights_total_bz_sum = weights_v_bz_sum + weights_c_bz_sum
        weights_total_bz_sum = weights_total_bz_sum/np.max(weights_total_bz_sum)

        #filter points outside of area
        lim = np.max(self.lattice.rlat)*limfactor
        dlim = lim*1.1

        filtered_weights_v = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_v_bz_sum) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
        x_v,y_v,weights_v_bz_sum = np.array(filtered_weights_v).T

        filtered_weights_c = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_c_bz_sum) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
        x_c,y_c,weights_c_bz_sum = np.array(filtered_weights_c).T

        filtered_weights_total = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_total_bz_sum) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
        x_total,y_total,weights_total_bz_sum = np.array(filtered_weights_total).T

        # Add contours of BZ
        ax.add_patch(BZ_Wigner_Seitz(self.lattice, color = 'black'))
        bx.add_patch(BZ_Wigner_Seitz(self.lattice, color = 'black'))
        cx.add_patch(BZ_Wigner_Seitz(self.lattice, color = 'black'))

        #plotting
        if mode == 'hexagon': 
            scale = kwargs.pop('scale',150)   
            s_v=ax.scatter(x_v,y_v,s=scale,marker='H',c=weights_v_bz_sum,rasterized=True,**kwargs)
            s_c=bx.scatter(x_c,y_c,s=scale,marker='H',c=weights_c_bz_sum,rasterized=True,**kwargs)
            s_total_1=cx.scatter(x_total,y_total,s=scale,marker='H',c=weights_v_bz_sum,rasterized=True,**kwargs)
            s_total_2=cx.scatter(x_total,y_total,s=scale,marker='H',c=weights_c_bz_sum,rasterized=True,**kwargs)
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
            bx.set_xlim(-lim,lim)
            bx.set_ylim(-lim,lim)
            cx.set_xlim(-lim,lim)
            cx.set_ylim(-lim,lim)

        elif mode == 'square': 
            scale = kwargs.pop('scale',1)
            if spin_pol == 'up':
               s=ax.scatter(x,y,s=scale,marker='s',c=weights_bz_sum_up,rasterized=True,**kwargs)
            elif spin_pol == 'dw':
               s=ax.scatter(x,y,s=scale,marker='s',c=weights_bz_sum_dw,rasterized=True,**kwargs)
            else:
               s=ax.scatter(x,y,s=scale,marker='s',c=weights_bz_sum,rasterized=True,**kwargs)
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
     
        elif mode == 'rbf':
            from scipy.interpolate import Rbf
            npts = kwargs.pop('npts',100)
            interp_method = kwargs.pop('interp_method','bicubic')


            rbfi_v = Rbf(x_v,y_v,weights_v_bz_sum,function='linear')
            x_v = y_v = np.linspace(-lim,lim,npts)
            weights_v_bz_sum = np.zeros([npts,npts])

            rbfi_c = Rbf(x_c,y_c,weights_c_bz_sum,function='linear')
            x_c = y_c = np.linspace(-lim,lim,npts)
            weights_c_bz_sum = np.zeros([npts,npts])

            rbfi_total = Rbf(x_total,y_total,weights_total_bz_sum,function='linear')
            x_total = y_total = np.linspace(-lim,lim,npts)
            weights_total_bz_sum = np.zeros([npts,npts])

            for col in range(npts): 
                weights_v_bz_sum[:,col] = rbfi_v(x_v,np.ones_like(x_v)*y_v[col])
                weights_c_bz_sum[:,col] = rbfi_c(x_c,np.ones_like(x_c)*y_c[col])
                weights_total_bz_sum[:,col] = rbfi_total(x_total,np.ones_like(x_total)*y_total[col])
                cmap_v = plt.cm.Blues
                cmap_c = plt.cm.Reds
                cmap_total = plt.cm.viridis
                s_v=ax.imshow(weights_v_bz_sum.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim], cmap = cmap_v, alpha = 1.0)
                s_c=bx.imshow(weights_c_bz_sum.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim], cmap = cmap_c, alpha = 1.0)
                s_total_1=cx.imshow(weights_v_bz_sum.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim], cmap = cmap_v, alpha = 1.0)
                s_total_2=cx.imshow(weights_c_bz_sum.T,interpolation=interp_method,extent=[-lim,lim,-lim,lim], cmap = cmap_c, alpha = 0.5)


        ax.set_title('Valence contribution')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        bx.set_title('Conduction contribution')
        bx.set_aspect('equal')
        bx.set_xticks([])
        bx.set_yticks([])

        cx.set_title('Total contribution')
        cx.set_aspect('equal')
        cx.set_xticks([])
        cx.set_yticks([])
      
        return ax,bx,cx,s_v,s_c
    
    def get_exciton_3D(self,excitons,f=None):
        """get data of the exciton in 2D"""
        weights = self.get_exciton_weights(excitons)
        #sum all the bands
        weights_bz_sum = np.sum(weights,axis=1)
        if f: weights_bz_sum = f(weights_bz_sum)

        kmesh_full, kmesh_idx = replicate_red_kmesh(self.lattice.red_kpoints,repx=range(-1,2),repy=range(-1,2))
        x,y,z = red_car(kmesh_full,self.lattice.rlat)[:,:3].T
        weights_bz_sum = weights_bz_sum[kmesh_idx]
        return x,y,z,weights_bz_sum
 
    def plot_exciton_3D_ax(self,ax,excitons,f=None,mode='hexagon',limfactor=0.8,**kwargs):
        """
        Plot the exciton weights in a 3D Brillouin zone
       
           Arguments:
            excitons -> list of exciton indexes to plot
            f -> function to apply to the exciton weights. Ex. f=log will compute the 
                 log of th weight to enhance the small contributions
            mode -> possible values are 'hexagon' to use hexagons as markers for the 
                    weights plot and 'rbf' to interpolate the weights using radial basis functions.
            limfactor -> factor of the lattice parameter to choose the limits of the plot 
            scale -> size of the markers
        """
        x,y,z,weights_bz_sum = self.get_exciton_3D(excitons,f=f)
        print(x,y,z,weights_bz_sum)


        #filter points outside of area
        lim = np.max(self.lattice.rlat)*limfactor
        dlim = lim*1.1
        filtered_weights = [[xi,yi,di] for xi,yi,di in zip(x,y,weights_bz_sum) if -dlim<xi and xi<dlim and -dlim<yi and yi<dlim]
        x,y,z,weights_bz_sum = np.array(filtered_weights).T

        #plotting
        if mode == 'hexagon': 
            scale = kwargs.pop('scale',1)
            ax.scatter(x,y,s=scale,marker='H',c=weights_bz_sum,rasterized=True,**kwargs)
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
        elif mode == 'rbf':
            from scipy.interpolate import Rbf
            npts = kwargs.pop('npts',100)
            interp_method = kwargs.pop('interp_method','bicubic')
            rbfi = Rbf(x,y,weights_bz_sum,function='linear')
            x = y = np.linspace(-lim,lim,npts)
            weights_bz_sum = np.zeros([npts,npts])
            for col in range(npts):
                weights_bz_sum[:,col] = rbfi(x,np.ones_like(x)*y[col])
            ax.imshow(weights_bz_sum,interpolation=interp_method)

        title = kwargs.pop('title',str(excitons))
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(BZ_Wigner_Seitz(self.lattice))
        return ax
    
    @add_fig_kwargs
    def plot_exciton_2D(self,excitons,f=None,**kwargs):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        self.plot_exciton_2D_ax(ax,excitons,f=f,**kwargs)
        return fig

    def plot_nbrightest_2D(self,emin=0,emax=10,estep=0.001,broad=0.1,
                           mode='rbf',scale=3,nrows=2,ncols=2,eps=1e-5):
        """
        Create a plot with chi and vertical bars for the brightest excitons
        Also plot the 2D wavefunctions of the brightest excitons.

          Arguments:
            emin,emax -> minimum and maximum energy range to plot chi
            estep -> energy step to plot chi
            broad -> broadening of the exciton peaks
            mode -> possible values are 'hexagon' to use hexagons as markers for the 
                    weights plot and 'rbf' to interpolate the weights using radial basis functions.
            scale -> size of the markers
            nrows,ncols -> number of rows and colums for the 2D plots (default: 2x2)
            eps -> threshold to find degenerate states
        """
        import matplotlib.pyplot as plt
        figexc = plt.figure()
        n_brightest = nrows*ncols
        figchi = self.plot_chi(emin=emin,emax=emax,estep=estep,broad=broad,n_brightest=n_brightest,show=False)
        #plot vertical bar on the brightest excitons
        exc_e,exc_i = self.get_sorted()
        sorted_exc = sorted(exc_i[:n_brightest],key = lambda x: x[1])
        for n,(i,idx) in enumerate(sorted_exc):
            ax = figexc.add_subplot(nrows,ncols,n+1)
            excitons = self.get_degenerate(idx,eps)
            self.plot_exciton_2D_ax(ax,excitons,scale=scale,mode=mode)
        return figchi,figexc

    def get_exciton_bs(self,energies_db,path,excitons,size=1,space='bands',f=None,debug=False):
        """
        Get a YambopyBandstructure object with the exciton band-structure
        
            Arguments:
            ax          -> axis extance of matplotlib to add the plot to
            lattice     -> Lattice database
            energies_db -> Energies database, can be either a SaveDB or QPDB
            path        -> Path in the brillouin zone
        """
        from qepy.lattice import Path
        if not isinstance(path,Path): 
            raise ValueError('Path argument must be a instance of Path. Got %s instead'%type(path))
    
        if space == 'bands':
            if self.spin_pol=='no':
               bands_kpoints, energies, weights = self.exciton_bs(energies_db, path.kpoints, excitons, debug)
               nkpoints = len(bands_kpoints)
               plot_energies = energies[:,self.start_band:self.mband]
               plot_weights  = weights[:,self.start_band:self.mband]
        #    elif spin_pol=='pol':
               
        else:
            raise NotImplementedError('TODO')
            eh_size = len(self.unique_vbands)*len(self.unique_cbands)
            nkpoints = len(bands_kpoints)
            plot_energies = np.zeros([nkpoints,eh_size])
            plot_weights = np.zeros([nkpoints,eh_size])
            for eh,(v,c) in enumerate(product(self.unique_vbands,self.unique_cbands)):
                plot_energies[:,eh] = energies[:,c]-energies[:,v]
                plot_weights[:,eh] = weights[:,c] 

        if f: plot_weights = f(plot_weights)
        size *= 1.0/np.max(plot_weights)
        ybs = YambopyBandStructure(plot_energies, bands_kpoints, weights=plot_weights, kpath=path, size=size)
        return ybs


    def plot_exciton_bs_ax(self,ax,energies_db,path,excitons,size=1,space='bands',f=None,debug=None):
        ybs = self.get_exciton_bs(energies_db,path,excitons,size=size,space=space,f=f,debug=debug)
        return ybs.plot_ax(ax) 

    @add_fig_kwargs
    def plot_exciton_bs(self,energies_db,path,excitons,size=1,space='bands',f=None,debug=False,**kwargs):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        self.plot_exciton_bs_ax(ax,energies_db,path,excitons,size=size,space=space,f=f,debug=debug)
        return fig

    def interpolate(self,energies,path,excitons,lpratio=5,f=None,size=1,verbose=True,**kwargs):
        """ Interpolate exciton bandstructure using SKW interpolation from Abipy
            This function is still with some bugs...
            for instance, kpoints_indexes should be read from savedb

        """
        from abipy.core.skw import SkwInterpolator

        if verbose:
            print("This interpolation is provided by the SKW interpolator implemented in Abipy")

        lattice = self.lattice
        cell = (lattice.lat, lattice.red_atomic_positions, lattice.atomic_numbers)
        nelect = 0
        # Here there is something strange...

        fermie = kwargs.pop('fermie',0)
        ##
        symrel = [sym for sym,trev in zip(lattice.sym_rec_red,lattice.time_rev_list) if trev==False ]
        time_rev = True
 
        weights = self.get_exciton_weights(excitons)
        weights = weights[:,self.start_band:self.mband]

        if f: weights = f(weights)
        size *= 1.0/np.max(weights)
        ibz_nkpoints = max(lattice.kpoints_indexes)+1

        #kpoints = lattice.red_kpoints   This is not needed, why is here? To be removed

        #map from bz -> ibz:
        # bug here? it is self.mband, but why? Start counting at zero?
        ibz_weights = np.zeros([ibz_nkpoints,self.mband-self.start_band]) 
        ibz_kpoints = np.zeros([ibz_nkpoints,3])

        # Kpoints indexes must be read from a SAVEDB Class
        k1,k2,k3 = energies.expand_kpts()
        kpoints_indexes = k2

        # Fijar este error
        for idx_bz,idx_ibz in enumerate(kpoints_indexes):
            ibz_weights[idx_ibz,:] = weights[idx_bz,:] 
            ibz_kpoints[idx_ibz] = lattice.red_kpoints[idx_bz]

        #get eigenvalues along the path
        if isinstance(energies,(YamboSaveDB,YamboElectronsDB)):
            ibz_energies = energies.eigenvalues[0,:,self.start_band:self.mband]
        elif isinstance(energies,YamboQPDB):
            ibz_energies = energies.eigenvalues_qp # to be done for spin-UP channel
        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))

        #interpolate energies
        na = np.newaxis
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_energies[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        energies = skw.interp_kpts(kpoints_path).eigens
        #interpolate weights
        na = np.newaxis
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_weights[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        exc_weights = skw.interp_kpts(kpoints_path).eigens

        #create band-structure object
        exc_bands = YambopyBandStructure(energies[0],kpoints_path,kpath=path,weights=exc_weights[0],size=size,**kwargs)
        #exc_bands.set_fermi(self.nvbands)

        return exc_bands

    def interpolate_finiteq(self,energies,kindx,path,excitons,lpratio=5,f=None,size=None,verbose=True,**kwargs):
        """ Interpolate exciton bandstructure using SKW interpolation from Abipy
            This function is still with some bugs...
            for instance, kpoints_indexes should be read from savedb

        """
        from abipy.core.skw import SkwInterpolator

        if verbose:
            print("This interpolation is provided by the SKW interpolator implemented in Abipy")

        lattice = self.lattice
        cell = (lattice.lat, lattice.red_atomic_positions, lattice.atomic_numbers)
        nelect = 0
        # Here there is something strange...

        fermie = kwargs.pop('fermie',0)
        ##
        symrel = [sym for sym,trev in zip(lattice.sym_rec_red,lattice.time_rev_list) if trev==False ]
        time_rev = True

        print('Q = ', self.Qpt)   # Check Q convention

        weights_v, weights_c = self.get_exciton_weights_finiteq(excitons,self.Qpt,kindx)

        print(weights_c)

        size_v = 0.0 
        size_c = 0.0

        size_v = np.max(weights_v)
        size_c = np.max(weights_c)

        print(size_v,size_c)

        ibz_nkpoints = max(lattice.kpoints_indexes)+1

        #kpoints = lattice.red_kpoints   This is not needed, why is here? To be removed

        #map from bz -> ibz:
        # bug here? it is self.mband, but why? Start counting at zero?

        ibz_weights_v = np.zeros([ibz_nkpoints,self.nbands]) 
        ibz_weights_c = np.zeros([ibz_nkpoints,self.nbands])
        ibz_kpoints = np.zeros([ibz_nkpoints,3]) 

        # Kpoints indexes must be read from a SAVEDB Class
        k1,k2,k3 = energies.expand_kpts()
        kpoints_indexes = k2

        # Fijar este error


        vmax = np.zeros([self.nkpoints,self.nbands])
        cmax = np.zeros([self.nkpoints,self.nbands])

        for idx_bz,idx_ibz in enumerate(kpoints_indexes):

            if idx_ibz == 0:
               ibz_weights_v[idx_ibz,:] = weights_v[idx_bz,:]
               ibz_weights_c[idx_ibz,:] = weights_c[idx_bz,:]
               print('A',idx_ibz,idx_bz)

            if idx_ibz != 0:
              
               if idx_ibz != kpoints_indexes[idx_bz-1]:

                  ibz_weights_v[idx_ibz,:] = weights_v[idx_bz,:]
                  ibz_weights_c[idx_ibz,:] = weights_c[idx_bz,:]
                  vmax[idx_bz,:] = weights_v[idx_bz,:]
                  cmax[idx_bz,:] = weights_c[idx_bz,:]
                  print('B',idx_ibz,idx_bz) 

               if idx_ibz == kpoints_indexes[idx_bz-1]:
                  
                  for v in range(self.nvbands):
                      vmax[idx_bz,v] = max(weights_v[idx_bz,v],vmax[idx_bz - 1,v])
                      ibz_weights_v[idx_ibz,v] = vmax[idx_bz,v]

                  for c in range(self.nvbands,self.nbands):
                      cmax[idx_bz,c] = max(weights_c[idx_bz,c],cmax[idx_bz - 1,c])
                      ibz_weights_c[idx_ibz,c] = cmax[idx_bz,c]
                  
                  print('C',idx_ibz,idx_bz)

            ibz_kpoints[idx_ibz] = lattice.red_kpoints[idx_bz]    
           
        #get eigenvalues along the path
        if isinstance(energies,(YamboSaveDB,YamboElectronsDB)):
            ibz_energies = energies.eigenvalues[0,:,self.start_band:self.mband]
        elif isinstance(energies,YamboQPDB):
            ibz_energies = energies.eigenvalues_qp # to be done for spin-UP channel
        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))

        #interpolate energies
        na = np.newaxis
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_energies[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        energies = skw.interp_kpts(kpoints_path).eigens

        #interpolate weights_c
        na = np.newaxis
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_weights_c[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        exc_weights_c = skw.interp_kpts(kpoints_path).eigens

        #interpolate weights_v
        na = np.newaxis
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_weights_v[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        exc_weights_v = skw.interp_kpts(kpoints_path).eigens

        # Hay que cuadrar las bandas 
        #create band-structure object
        exc_bands_v = YambopyBandStructure(energies[0],kpoints_path,kpath=path,weights=exc_weights_v[0],size=size_v,**kwargs)
        exc_bands_c = YambopyBandStructure(energies[0],kpoints_path,kpath=path,weights=exc_weights_c[0],size=size_c,**kwargs) 
        #exc_bands.set_fermi(self.nvbands)
       

        return exc_bands_v, exc_bands_c, size_v, size_c



    def interpolate_transitions(self,energies,path,excitons,lpratio=5,f=None,size=1,verbose=True,**kwargs):
        """ Interpolate exciton bandstructure using SKW interpolation from Abipy
        """
        from abipy.core.skw import SkwInterpolator

        if verbose:
            print("This interpolation is provided by the SKW interpolator implemented in Abipy")

        lattice = self.lattice
        cell = (lattice.lat, lattice.red_atomic_positions, lattice.atomic_numbers)
        nelect = 0
        # Here there is something strange...
        fermie = kwargs.pop('fermie',0)
        ##
        symrel = [sym for sym,trev in zip(lattice.sym_rec_red,lattice.time_rev_list) if trev==False ]
        time_rev = True

        #vmin, vmax = self.unique_vbands[0], self.unique_vbands[1]
        #cmin, cmax = self.unique_cbands[0], self.unique_cbands[1]

        transitions = self.get_exciton_transitions(excitons)
        transitions = transitions[:,:,:]

        ibz_nkpoints = max(lattice.kpoints_indexes)+1
        kpoints = lattice.red_kpoints

        #map from bz -> ibz:
        ibz_transitions = np.zeros([ibz_nkpoints,self.nvbands,self.ncbands])
        ibz_kpoints = np.zeros([ibz_nkpoints,3])
        for idx_bz,idx_ibz in enumerate(lattice.kpoints_indexes):
            ibz_transitions[idx_ibz,:,:] = transitions[idx_bz,:,:] 
            ibz_kpoints[idx_ibz] = lattice.red_kpoints[idx_bz]

        #get eigenvalues along the path
        if isinstance(energies,(YamboSaveDB,YamboElectronsDB)):
            ibz_energies = energies.eigenvalues[:,self.start_band:self.mband]
        elif isinstance(energies,YamboQPDB):
            ibz_energies = energies.eigenvalues_qp
        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))

        #interpolate energies
        na = np.newaxis
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_energies[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        energies = skw.interp_kpts(kpoints_path).eigens
     
        #interpolate transitions
        na = np.newaxis
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_transitions[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        exc_transitions = skw.interp_kpts(kpoints_path).eigens

        print(exc_transitions.shape)
        exit()

        #create band-structure object
        exc_bands = YambopyBandStructure(energies[0],kpoints_path,kpath=path,weights=exc_weights[0],size=size,**kwargs)
        exc_bands.set_fermi(self.nvbands)

        return exc_transitions


    def interpolate_spin(self,energies,spin_proj,path,excitons,lpratio=5,f=None,size=1,verbose=True,**kwargs):
        """ Interpolate exciton bandstructure using SKW interpolation from Abipy
        """
        from abipy.core.skw import SkwInterpolator

        if verbose:
            print("This interpolation is provided by the SKW interpolator implemented in Abipy")

        lattice = self.lattice
        cell = (lattice.lat, lattice.red_atomic_positions, lattice.atomic_numbers)
        nelect = 0
        # Here there is something strange...

        fermie = kwargs.pop('fermie',0)
        ##
        symrel = [sym for sym,trev in zip(lattice.sym_rec_red,lattice.time_rev_list) if trev==False ]
        time_rev = True
 
        weights = self.get_exciton_weights(excitons)
        weights = weights[:,self.start_band:self.mband]
        if f: weights = f(weights)
        size *= 1.0/np.max(weights)
        ibz_nkpoints = max(lattice.kpoints_indexes)+1
        kpoints = lattice.red_kpoints

        #map from bz -> ibz:
        print("ibz_nkpoints")
        print(ibz_nkpoints)
        print("weights.shape")
        print(weights.shape)
        print(self.unique_vbands)
        print(self.unique_cbands)
        v_1 = self.unique_vbands[ 0]
        v_2 = self.unique_cbands[-1] + 1
        #exit()
        ibz_weights = np.zeros([ibz_nkpoints,self.nbands])
        ibz_kpoints = np.zeros([ibz_nkpoints,3])
        ibz_spin    = np.zeros([ibz_nkpoints,self.nbands])
        for idx_bz,idx_ibz in enumerate(lattice.kpoints_indexes):
            ibz_weights[idx_ibz,:] = weights[idx_bz,:] 
            ibz_kpoints[idx_ibz]   = lattice.red_kpoints[idx_bz]
            ibz_spin[idx_ibz,:]    = spin_proj[idx_bz,v_1:v_2]
        #get eigenvalues along the path
        if isinstance(energies,(YamboSaveDB,YamboElectronsDB)):
            ibz_energies = energies.eigenvalues[:,self.start_band:self.mband]
        elif isinstance(energies,YamboQPDB):
            ibz_energies = energies.eigenvalues_qp
        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))

        #interpolate energies
        na = np.newaxis
        print("na")
        print(na)
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_energies[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        energies = skw.interp_kpts(kpoints_path).eigens
     
        #interpolate weights
        na = np.newaxis
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_weights[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        exc_weights = skw.interp_kpts(kpoints_path).eigens

        #interpolate spin projection
        na = np.newaxis
        print("na")
        print(na)
        skw = SkwInterpolator(lpratio,ibz_kpoints,ibz_spin[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        spin_inter   = skw.interp_kpts(kpoints_path).eigens
        print("spin_inter")
        print(spin_inter)

        #create band-structure object
        exc_bands = YambopyBandStructure(energies[0],kpoints_path,kpath=path,weights=exc_weights[0],spin_proj=spin_inter[0],size=size,**kwargs)
        exc_bands.set_fermi(self.nvbands)

        return exc_bands
 
    def get_amplitudes_phases(self,excitons=(0,),repx=list(range(1)),repy=list(range(1)),repz=list(range(1))):
        """ get the excitonic amplitudes and phases
        """
        if self.eigenvectors is None:
            raise ValueError('This database does not contain Excitonic states,'
                             'please re-run the yambo BSE calculation with the WRbsWF option in the input file.')
        if isinstance(excitons, int):
            excitons = (excitons,)
       
        car_kpoints = self.lattice.car_kpoints
        nkpoints = len(car_kpoints)
        print(nkpoints)
        amplitudes = np.zeros([nkpoints])
        phases     = np.zeros([nkpoints],dtype=np.complex64)
        for exciton in excitons:
            #the the eigenstate
            eivec = self.eigenvectors[exciton-1]
           
            total = 0
            for eh,kvc in enumerate(self.table):
                ikbz, v, c = kvc-1
                Acvk = eivec[eh]
                phases[ikbz]     += Acvk
                amplitudes[ikbz] += np.abs(Acvk)

        #replicate kmesh
        red_kmesh,kindx = replicate_red_kmesh(self.lattice.red_kpoints,repx=repx,repy=repy,repz=repz)
        car_kpoints = red_car(red_kmesh,self.lattice.rlat)

        return car_kpoints, amplitudes[kindx], np.angle(phases)[kindx]

    def get_chi(self,dipoles=None,dir=0,emin=0,emax=10,estep=0.01,broad=0.1,q0norm=1e-5, nexcitons='all',spin_degen=2,verbose=0,**kwargs):
        """
        Calculate the dielectric response function using excitonic states
        """
        if nexcitons == 'all': nexcitons = self.nexcitons

        #energy range
        w = np.arange(emin,emax,estep,dtype=np.float32)
        nenergies = len(w)
        
        if verbose:
            print("energy range: %lf -> +%lf -> %lf "%(emin,estep,emax))
            print("energy steps: %lf"%nenergies)

        #initialize the susceptibility intensity
        chi = np.zeros([len(w)],dtype=np.complex64)

        if dipoles is None:
            #get dipole
            EL1 = self.l_residual
            EL2 = self.r_residual
        else:
            #calculate exciton-light coupling
            if verbose: print("calculate exciton-light coupling")
            EL1,EL2 = self.project1(dipoles.dipoles[:,dir],nexcitons) 

        if isinstance(broad,float): broad = [broad]*nexcitons

        if isinstance(broad,tuple): 
            broad_slope = broad[1]-broad[0]
            min_exciton = np.min(self.eigenvalues.real)
            broad = [ broad[0]+(es-min_exciton)*broad_slope for es in self.eigenvalues[:nexcitons].real]

        if "gaussian" in broad or "lorentzian" in broad:
            i = broad.find(":")
            if i != -1:
                value, eunit = broad[i+1:].split()
                if eunit == "eV": sigma = float(value)
                else: raise ValueError('Unknown unit %s'%eunit)

            f = gaussian if "gaussian" in broad else lorentzian
            broad = np.zeros([nexcitons])
            for s in range(nexcitons):
                es = self.eigenvalues[s].real
                broad += f(self.eigenvalues.real,es,sigma)
            broad = 0.1*broad/nexcitons

        #iterate over the excitonic states
        for s in range(nexcitons):
            #get exciton energy
            es = self.eigenvalues[s]
 
            #calculate the green's functions
            G1 = -1/(   w - es + broad[s]*I)
            G2 = -1/( - w - es - broad[s]*I)

            r = EL1[s]*EL2[s]
            chi += r*G1 + r*G2

        #dimensional factors
        if not self.Qpt==1: q0norm = 2*np.pi*np.linalg.norm(self.car_qpoint)
        if self.q_cutoff is not None: q0norm = self.q_cutoff

        d3k_factor = self.lattice.rlat_vol/self.lattice.nkpoints
        cofactor = ha2ev*spin_degen/(2*np.pi)**3 * d3k_factor * (4*np.pi)  / q0norm**2
        
        chi = 1. + chi*cofactor #We are actually computing the epsilon, not the chi.

        return w,chi

    def plot_chi_ax(self,ax,reim='im',n_brightest=-1,**kwargs):
        """Plot chi on a matplotlib axes"""
        w,chi = self.get_chi(**kwargs)
        #cleanup kwargs variables
        cleanup_vars = ['dipoles','dir','emin','emax','estep','broad',
                        'q0norm','nexcitons','spin_degen','verbose']
        for var in cleanup_vars: kwargs.pop(var,None)
        if 're' in reim: ax.plot(w,chi.real,**kwargs)
        if 'im' in reim: ax.plot(w,chi.imag,**kwargs)
        ax.set_ylabel('$Im(\chi(\omega))$')
        ax.set_xlabel('Energy (eV)')
        #plot vertical bar on the brightest excitons
        if n_brightest>-1:
            exc_e,exc_i = self.get_sorted()
            for i,idx in exc_i[:n_brightest]:
                exciton_energy,idx = exc_e[idx]
                ax.axvline(exciton_energy,c='k')
                ax.text(exciton_energy,0.1,idx,rotation=90)
        return w,chi

    @add_fig_kwargs
    def plot_chi(self,n_brightest=-1,**kwargs):
        """Produce a figure with chi"""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        self.plot_chi_ax(ax,n_brightest=n_brightest,**kwargs)
        return fig

    def save_chi(self,filename,**kwargs):
        """Compute chi and dump it to file"""
        w,chi = self.get_chi(**kwargs)
        np.savetxt(filename,np.array([w,chi.imag,chi.real]).T)


    ##########################################
    #  ARPES finite-q #
    ##########################################


    def arpes_intensity(self,energies_db,path,excitons,ax):   #,size=1,space='bands',f=None,debug=False): later on
        size=1 # luego lo ponemos como input variable 
        n_excitons = len(excitons)
        #
        kpath   = path
        # kpoints IBZ
        kpoints = self.lattice.red_kpoints
        # array of high symmetry k-points
        path    = np.array(path.kpoints)

        # Expansion of IBZ kpoints to Path kpoints
        rep = list(range(-1,2))
        kpoints_rep, kpoints_idx_rep = replicate_red_kmesh(kpoints,repx=rep,repy=rep,repz=rep)
        band_indexes = get_path(kpoints_rep,path)
        band_kpoints = np.array(kpoints_rep[band_indexes])
        band_indexes = kpoints_idx_rep[band_indexes]

        # Eigenvalues Full BZ
        # Dimension nk_fbz x nbands
        energies = energies_db.eigenvalues[self.lattice.kpoints_indexes]

        # Calculate omega
        # omega_vk,lambda = e_(v,k-q) + omega_(lambda,q) only for q=0
        '''
        omega_vkl = np.zeros([self.nkpoints, self.nvbands,n_excitons])
        for i_l,exciton in enumerate(excitons):
            for i_k in range(self.nkpoints):
                for i_v in range(self.nvbands):
                    i_v2 = self.unique_vbands[i_v]
                    # omega_vk,lambda      = e_(v,k-q) + omega_(lambda,q)
                    omega_vkl[i_k,i_v,i_l] = energies[i_k,i_v2] + self.eigenvalues.real[exciton-1]

        '''
        omega_vkl = self.calculate_omega(energies,excitons)
        rho       = self.calculate_rho(excitons)
        # Calculate rho's
        # rho_vk = Sum_{c} |A_cvk|^2
#        rho = np.zeros([self.nkpoints, self.nvbands, n_excitons])


#        for i_exc, exciton in enumerate(excitons):
#            # get the eigenstate
#            eivec = self.eigenvectors[exciton-1]
#            for t,kvc in enumerate(self.table):
#                k,v,c = kvc[0:3]-1    # This is bug's source between yambo 4.4 and 5.0 check all this part of the class
#                i_v = v - self.nvbands                    # index de VB bands (start at 0)
#                i_c = c - self.ncbands - self.nvbands     # index de CB bands (start at 0)
#                rho[k,i_v,i_exc] += abs2(eivec[t])

        # Eigenvalues Path contains in Full BZ
        energies_path  = energies[band_indexes]
        rho_path       = rho[band_indexes]
        omega_vkl_path = omega_vkl[band_indexes]

        #make top valence band to be zero
        energies_path -= max(energies_path[:,max(self.unique_vbands)])

        plot_energies = energies_path[:,self.start_band:self.mband]
  
        # LDA or GW band structure
        ybs_bands = YambopyBandStructure(plot_energies, band_kpoints, kpath=kpath)


        # Intensity Plot
        print('shape energies_path')
        nkpoints_path=energies_path.shape[0]
        #exit()
        # Intensity histogram
        # I(k,omega_band)
        omega_band = np.arange(0.0,7.0,0.01)
        n_omegas = len(omega_band)
        Intensity = np.zeros([n_omegas,nkpoints_path]) 
        Im = 1.0j
           #for i_o in range(n_omegas):

        for i_o in range(n_omegas):
            for i_k in range(nkpoints_path):
                for i_v in range(self.nvbands):
                    for i_exc in range(n_excitons):
                        delta = 1.0/( omega_band[i_o] - omega_vkl_path[i_k,i_v,i_exc] + Im*0.2 )
                        Intensity[i_o,i_k] += rho_path[i_k,i_v,i_exc]*delta.imag

        distances = [0]
        distance = 0
        for nk in range(1,nkpoints_path):
            distance += np.linalg.norm(band_kpoints[nk]-band_kpoints[nk-1])
            distances.append(distance)
        distances = np.array(distances)
        X, Y = np.meshgrid(distances, omega_band)
        import matplotlib.pyplot as plt
        #plt.imshow(Intensity, interpolation='bilinear',cmap='viridis_r')
        plt.pcolor(X, Y, Intensity,cmap='viridis_r',shading='auto')
        # Excitonic Band Structure
        for i_v in range(self.nvbands):
            for i_exc in range(n_excitons):
                plt.plot(distances,omega_vkl_path[:,i_v,i_exc],color='w',lw=0.5) 
        # Electronic Band Structure
       
        for i_b in range(energies_db.nbands):
            plt.plot(distances,energies_path[:,i_b],lw=1.0,color='r')
        plt.xlim((distances[0],distances[-1]))
        plt.ylim((-5,10))
        plt.show()
        exit()

        # ARPES band structure
        ybs_omega = []
        for i_exc in range(n_excitons):
            plot_omega    = omega_vkl_path[:,:,i_exc]
            plot_rho      = rho_path[:,:,i_exc]
            size *= 1.0/np.max(plot_rho)
            ybs_omega.append( YambopyBandStructure(plot_omega, band_kpoints, weights=plot_rho, kpath=kpath, size=size) )

        # Plot bands
        ybs_bands.plot_ax(ax,color_bands='black',lw_label=2)

        for ybs in ybs_omega:
            ybs.plot_ax(ax,color_bands='black',lw_label=0.1)

        return rho

    def calculate_omega_finiteq(self,energies,energies_db,kindx, eigenvec_q, eigenval_q, Nexcitons):
        """ Calculate:
            omega_vk,lambda,q = e_(v,k-q) + omega_(lambda,q)
        """

        omega_vkl_q = np.zeros([self.nkpoints, self.nvbands, Nexcitons, self.nqpoints])

        step_omega_time = Nexcitons // 20  # 5% of total_loops

        start_time = time.time()

        print("")
        print("")
        print("omega_q calculation")
        print("")
        print("Progress: 0%, Estimated time remaining: Calculating...")


        if isinstance(energies_db,(YamboSaveDB,YamboElectronsDB)):

           for i_exc in range(Nexcitons):

               for iq in range(self.nqpoints):

                   for i_k in range(self.nkpoints):

                       for i_v in range(self.nvbands):

                           i_v2 = self.unique_vbands[i_v]
                           k_c = kindx.qindx_X[iq,i_k,0] - 1
                           omega_vkl_q[i_k,i_v,i_exc,iq] = energies[k_c,i_v2] + eigenval_q.real[iq,i_exc]

           if (i_exc + 1) % step_omega_time == 0 or i_exc == Nexcitons - 1:
              elapsed_time = time.time() - start_time
              progress = ((i_exc + 1) / Nexcitons) * 100
              estimated_total_time = (elapsed_time / (i_exc + 1)) * Nexcitons
              estimated_time_remaining = estimated_total_time - elapsed_time

              # Convert times to hours, minutes, and seconds
              elapsed_hours, elapsed_minutes, elapsed_seconds = self.seconds_to_hms(elapsed_time)
              remaining_hours, remaining_minutes, remaining_seconds = self.seconds_to_hms(estimated_time_remaining)

              print(f"Progress: {progress:.1f}%, Estimated time remaining: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s\n")

        elif isinstance(energies_db,YamboQPDB):   # To work correctly, the number of valence bands in BSEBands and the number of valence bands in the GW calculation must be the same.

             for i_exc in range(Nexcitons):

                 for iq in range(self.nqpoints):

                     for i_k in range(self.nkpoints):

                         for i_v in range(self.nvbands):
 
                             k_c = kindx.qindx_X[iq,i_k,0] - 1
                             omega_vkl_q[i_k,i_v,i_exc,iq] = energies[k_c,i_v] + eigenval_q.real[iq,i_exc]

             if (i_exc + 1) % step_omega_time == 0 or i_exc == Nexcitons - 1:
                elapsed_time = time.time() - start_time
                progress = ((i_exc + 1) / Nexcitons) * 100
                estimated_total_time = (elapsed_time / (i_exc + 1)) * Nexcitons
                estimated_time_remaining = estimated_total_time - elapsed_time

                # Convert times to hours, minutes, and seconds
                elapsed_hours, elapsed_minutes, elapsed_seconds = self.seconds_to_hms(elapsed_time)
                remaining_hours, remaining_minutes, remaining_seconds = self.seconds_to_hms(estimated_time_remaining)

                print(f"Progress: {progress:.1f}%, Estimated time remaining: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s\n")

        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))

        # Final time
        end_time = time.time()
        total_time = end_time - start_time
        total_hours, total_minutes, total_seconds = self.seconds_to_hms(total_time)

        np.save('omega_vkl_q.npy',omega_vkl_q)

        print(f"Total time taken: {total_hours}h {total_minutes}m {total_seconds}s\n")
        print('')
        print('omega_q computed')

        return omega_vkl_q


    def calculate_rho_finiteq(self, kindx, eigenvec_q, eigenval_q, Nexcitons):
        """ Calculate:
            rho_vkl = Sum_{q,c} |A_cvk,q,l|^2
        """
         
        rho_q = np.zeros([self.nkpoints, self.nvbands, Nexcitons, self.nqpoints])

        step_rho_time = Nexcitons // 20  # 5% of total_loops

        start_time = time.time()

        print("")
        print("")
        print("rho_q calculation")
        print("")
        print("Progress: 0%, Estimated time remaining: Calculating...")

        for i_exc in range(Nexcitons):

            for iq in range(self.nqpoints):

                for t,kvc in enumerate(self.table):
                    k,v,c = kvc[0:3]-1    
                    i_v = v - self.unique_vbands[0] 
                    rho_q[k,i_v,i_exc,iq] += abs2(eigenvec_q[t,iq,i_exc])

            if (i_exc + 1) % step_rho_time == 0 or i_exc == Nexcitons - 1:
               elapsed_time = time.time() - start_time
               progress = ((i_exc + 1) / Nexcitons) * 100
               estimated_total_time = (elapsed_time / (i_exc + 1)) * Nexcitons
               estimated_time_remaining = estimated_total_time - elapsed_time

               # Convert times to hours, minutes, and seconds
               elapsed_hours, elapsed_minutes, elapsed_seconds = self.seconds_to_hms(elapsed_time)
               remaining_hours, remaining_minutes, remaining_seconds = self.seconds_to_hms(estimated_time_remaining)

               print(f"Progress: {progress:.1f}%, Estimated time remaining: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s\n")

        np.save('rho_q.npy', rho_q)

        # Final time
        end_time = time.time()
        total_time = end_time - start_time
        total_hours, total_minutes, total_seconds = self.seconds_to_hms(total_time)

        print(f"Total time taken: {total_hours}h {total_minutes}m {total_seconds}s\n")
        print('')
        print('rho_q computed')

        return rho_q

    def Boltz_dist(self, Time, eigenvec_q, eigenval_q, Nexcitons):

        Btz_d = np.zeros([self.nqpoints,Nexcitons])

        k = 8.6173e-5

        Temp = 10000*np.exp(-(Time)/50)

        print('La temperatura es = ', Temp, 'K')
        print('La Beta es = ', (1)/(k*Temp))

        sum_den = 0.0

        for i_exc in range(Nexcitons):
            for iq in range(self.nqpoints):
                sum_den += np.exp( - (eigenval_q.real[iq,i_exc]) / (k*Temp) )

        for i_exc in range(Nexcitons):
            for iq in range(self.nqpoints):
                Btz_d[iq,i_exc] = ( ( np.exp( - (eigenval_q.real[iq,i_exc]) / (k*Temp) ) ) / ( sum_den ) )

        return Btz_d

    def Gauss_func(self, sigma_omega, sigma_momentum, omega_in, qmod_in, w_0, qmod_0):

        Gaussian = ( (1.0) / (sigma_omega  * np.sqrt(2.0 * np.pi) ) )*( np.exp( (-0.5)*( ( (omega_in - w_0)/(sigma_omega) )**2 ) ) )

        return Gaussian


    def Gauss_dist(self,save, sigma_omega, sigma_momentum, eigenval_q, iq_fixed, Time, Pump_energy, Nexcitons):

        Gauss_dis = np.zeros([self.nqpoints,Nexcitons])

        Exc_dif = Pump_energy - np.min(eigenval_q.real[0,:])

        w_0 = Pump_energy - (Exc_dif*Time)/(100)

        qmod_0 = np.sqrt( abs2(save.red_kpoints[iq_fixed][0]) + abs2(save.red_kpoints[iq_fixed][1]) + abs2(save.red_kpoints[iq_fixed][2]) ) 

        if Time < 5:

           for i_exc in range(Nexcitons):

                for iq in range(self.nqpoints):

                       qmod_in = np.sqrt( abs2(save.red_kpoints[iq][0]) + abs2(save.red_kpoints[iq][1]) + abs2(save.red_kpoints[iq][2]) ) 


                       if iq == iq_fixed:

                          Gauss_dis[iq,i_exc] = self.Gauss_func(sigma_omega,sigma_momentum,eigenval_q.real[iq,i_exc],qmod_in,w_0,qmod_0)

                       else:

                          Gauss_dis[iq,i_exc] = 0.0

   
        if Time >= 5:

           for i_exc in range(Nexcitons):

               for iq in range(self.nqpoints):

                   qmod_in = np.sqrt( abs2(save.red_kpoints[iq][0]) + abs2(save.red_kpoints[iq][1]) + abs2(save.red_kpoints[iq][2]) ) 


                   Gauss_dis[iq,i_exc] = self.Gauss_func(sigma_omega,sigma_momentum,eigenval_q.real[iq,i_exc],qmod_in,w_0,qmod_0)


        Gauss_renorm = 0.0

        for i_exc in range(Nexcitons):
            for iq in range(self.nqpoints):
              Gauss_renorm += Gauss_dis[iq,i_exc]

        Gauss_dis = Gauss_dis/Gauss_renorm

        return Gauss_dis

    def GBdist(self, Time, Gauss_dis, Btz_d, Nexcitons):

        GBdist = np.zeros([self.nqpoints,Nexcitons])

        x_values = np.linspace(-50, 50, 100)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def smooth_step_function(x, threshold=0, slope=0.15):
            return sigmoid(slope * (x - threshold))

        Alpha = smooth_step_function(x_values)

        Signal_construction = smooth_step_function(x_values, threshold = -40, slope = 0.70)

        if Time < 100:

           for i_exc in range(Nexcitons):

               for iq in range(self.nqpoints):

                   if (Btz_d[iq,i_exc] / Gauss_dis[iq,i_exc]) > 1e3:

                      GBdist[iq,i_exc] = 0.0

                   else:

                      GBdist[iq,i_exc] = ((1.0 - Alpha[Time])*Gauss_dis[iq,i_exc] + Alpha[Time]*Btz_d[iq,i_exc])*Signal_construction[Time]

        if Time >= 100:

           for i_exc in range(Nexcitons):

               for iq in range(self.nqpoints):

                   GBdist[iq,i_exc] = Btz_d[iq,i_exc]

        return GBdist

    def eigenvec_val_q(self,Nexcitons,Restart = False):

        if Restart == False:

           step_eigenvalues = Nexcitons // 20  # 5% of total_loops

           start_time = time.time()

           print("")
           print("")
           print("Reading Q-excitons")
           print("")
           print("Progress: 0%, Estimated time remaining: Reading...")

           eigenval_q = np.zeros([self.nqpoints, Nexcitons], dtype = complex) 
           eigenvec_q = np.zeros([len(self.eigenvectors[0]), self.nqpoints, Nexcitons], dtype = complex) 

           for iq in range(self.nqpoints):

               yexc_list = self.from_db_file(self.lattice,filename='ndb.BS_diago_Q1',folder='yambo')[iq]     

               for ik in range(Nexcitons):
                   eigenval_q[iq,ik] = yexc_list.eigenvalues[ik]
  
                   for t in range(len(self.eigenvectors[0])):
                       eigenvec_q[t,iq,ik] = yexc_list.eigenvectors[ik,t]


           if (iq + 1) % step_eigenvalues == 0 or iq == self.nqpoints - 1:
              elapsed_time = time.time() - start_time
              progress = ((iq + 1) / self.nqpoints ) * 100
              estimated_total_time = (elapsed_time / (iq + 1)) * self.nqpoints 
              estimated_time_remaining = estimated_total_time - elapsed_time

              # Convert times to hours, minutes, and seconds
              elapsed_hours, elapsed_minutes, elapsed_seconds = self.seconds_to_hms(elapsed_time)
              remaining_hours, remaining_minutes, remaining_seconds = self.seconds_to_hms(estimated_time_remaining)

              print(f"Progress: {progress:.1f}%, Estimated time remaining: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s\n")

           np.save('eigenval_q.npy', eigenval_q)
           np.save('eigenvec_q.npy', eigenvec_q)

           # Final time
           end_time = time.time()
           total_time = end_time - start_time
           total_hours, total_minutes, total_seconds = self.seconds_to_hms(total_time)

           print(f"Total time taken: {total_hours}h {total_minutes}m {total_seconds}s\n")
           print('')
           print('rho_q computed')

        if Restart == True:

           eigenval_q = np.load('eigenval_q.npy')
           eigenvec_q = np.load('eigenvec_q.npy')

        return eigenval_q, eigenvec_q

    def tr_excitonDOS(self, lat, omega_1, omega_2, omega_step, sigma, eigenval_q, eigenvec_q, Time, Pump_energy, iq_fixed, Nexcitons):

        omega_band  = np.arange(omega_1,omega_2,omega_step)

        eDOS = np.zeros(len(omega_band))
        OccExcDOS = np.zeros(len(omega_band))

        sigma_omega = Time/300
        sigma_momentum = Time/30

        Gaussian = self.Gauss_dist(lat, sigma_omega, sigma_momentum, eigenval_q, iq_fixed, Time, Pump_energy, Nexcitons)
        Boltz = self.Boltz_dist(Time, eigenvec_q, eigenval_q, Nexcitons)
        GBdistribution = self.GBdist(Time, Gaussian, Boltz, Nexcitons)

        for w in range(len(omega_band)):
        
            for i_exc in range(Nexcitons):

                for iq in range(self.nqpoints):
          
                    eDOS[w] += self.Gauss_func(sigma, 1.0, eigenval_q[iq,i_exc].real, 0.0, omega_band[w], 0.0)
                    OccExcDOS[w] += self.Gauss_func(sigma, 1.0, eigenval_q[iq,i_exc].real, 0.0, omega_band[w], 0.0) * GBdistribution[iq,i_exc]
                    print(omega_band[w])


        return omega_band, eDOS, OccExcDOS

    def PE_distribution(self,ax,energies_db,Time,eigenval_q,eigenvec_q,iq_fixed,Pump_energy,Nexcitons,**kwargs):
        
        Time_dis_T = np.linspace(0, 350, 350)

        TempInset = 10000*np.exp(-(Time_dis_T)/50)

        x_values = np.linspace(-50, 50, 100)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def smooth_step_function(x, threshold=0, slope=0.15):
            return sigmoid(slope * (x - threshold))

        Alpha_dis = smooth_step_function(x_values)

        sigma_omega = Time/300
        sigma_momentum = Time/50

        Gaussian = self.Gauss_dist(energies_db, sigma_omega, sigma_momentum, eigenval_q, iq_fixed, Time, Pump_energy, Nexcitons)
        Boltz = self.Boltz_dist(Time, eigenvec_q, eigenval_q, Nexcitons)


        ax.plot(Time_dis_T,TempInset, color = 'darkorange', lw = 2, label = 'T(t)')

        ax.legend(loc = (0.70,0.95), frameon = False)

        ax.set_xlim(0.0,350.0)
        ax.set_ylim(0.0,10000)

        ax2 = ax.twinx()
        ax2.set_xlim(0.0,350.0)
        ax2.set_ylim(0.00,1.00)

        ax2.plot(x_values + 50,Alpha_dis, color = 'magenta', lw = 2, label = 'Alpha (t)')


        ax2.axvline(x = Time, linestyle = 'dashed', lw = 1, color = 'black')
        ax2.legend(loc = (0.70,0.90), frameon = False)
        
        GaussianPerc = '% Gaussian'
        BoltzmannPerc = '% Boltzmann'
        PumpEnergyPerc = '$\omega_{0}$ = '
        PumpEnergyPerc_time = '$\omega$(t) = '
        Time_perc = 'Time = '
        Temp_perc = 'Temp = '


        Exc_dif_Perc = Pump_energy - np.min(eigenval_q.real[0,:])

        w_0_Perc = Pump_energy - (Exc_dif_Perc*Time)/(100)

        if Time < 100:
           ax.text(248,8600,f"{((1-Alpha_dis[Time])*100):.3f} {GaussianPerc}")
           ax.text(248,8200,f"{(Alpha_dis[Time]*100):.3f} {BoltzmannPerc}")
           ax.text(248,7800,f"{PumpEnergyPerc} {Pump_energy:.3f} {'eV'}")
           ax.text(248,7400,f"{PumpEnergyPerc_time} {w_0_Perc:.3f} {'eV'}")
           ax.text(248,7000,f"{Temp_perc} {TempInset[Time]:.0f}")

        else:
           ax.text(248,8600,f"{0.0} {GaussianPerc}")
           ax.text(248,8200,f"{100} {BoltzmannPerc}")          
           ax.text(248,7800,f"{PumpEnergyPerc} {Pump_energy:.3f} {'eV'}")
           ax.text(248,7200,f"{PumpEnergyPerc_time} {(Pump_energy - Exc_dif_Perc):.3f} {'eV'}") 
           ax.text(248,7000,f"{Temp_perc} {TempInset[Time]:.0f}")

        ax.text(248,6600,f"{Time_perc} {Time}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature')
        ax2.set_ylabel('Distribution - Gaussian vs Boltzmann', rotation = 270, labelpad = 20)

        ax.spines["bottom"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        ax.tick_params(labelsize=12, top = 'on', right = 'on', direction = 'in')
        ax.tick_params(axis = 'x', pad = 5)
        ax.tick_params(axis = 'y', pad = 10)
        ax.yaxis.label.set_size(12)
        ax.xaxis.label.set_size(12)
        ax2.spines["bottom"].set_linewidth(1)
        ax2.spines["top"].set_linewidth(1)
        ax2.spines["right"].set_linewidth(1)
        ax2.spines["left"].set_linewidth(1)
        ax2.tick_params(labelsize=12, top = 'on', right = 'on', direction = 'in')
        ax2.tick_params(axis = 'x', pad = 5)
        ax2.tick_params(axis = 'y', pad = 10)
        ax2.yaxis.label.set_size(12)
        ax2.xaxis.label.set_size(12)
        
        ind = (0, 2000, 4000, 6000, 8000, 10000)

        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ind))
        ax.set_yticklabels(['0', '2000', '4000', '6000', '8000', '10000'])

        ind2 = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

        ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ind2))
        ax2.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        
        return 

    def Photoemission(self,energies_db,lat,kindx,path,ax,cx,bx,Time,eigenval_q,eigenvec_q,Pump_energy,Nexcitons,iq_fixed,lpratio=5,f=None,size=1,verbose=True,Restart=False,**kwargs):

        from abipy.core.skw import SkwInterpolator
        Im = 1.0j # Imaginary
        
        if verbose:
           print("This interpolation is provided by the SKW interpolator implemented in Abipy")

        # Options kwargs
        # Alignment of the Bands Top Valence
        fermie      = kwargs.pop('fermie',0)
        # Parameters ARPES Intensity

        omega_width = kwargs.pop('omega_width',0)
        omega_1     = kwargs.pop('omega_1',0)
        omega_2     = kwargs.pop('omega_2',0)
        omega_step  = kwargs.pop('omega_step',0)
        omega_band  = np.arange(omega_1,omega_2,omega_step)
        n_omegas = len(omega_band)
        sigma     = kwargs.pop('sigma',0)
        cmap_name   = kwargs.pop('cmap_name',0)
        scissor    = kwargs.pop('scissor',0)

        # Lattice and Symmetry Variables
        lattice = self.lattice
        cell = (lattice.lat, lattice.red_atomic_positions, lattice.atomic_numbers)
        symrel = [sym for sym,trev in zip(lattice.sym_rec_red,lattice.time_rev_list) if trev==False ]
        time_rev = True
        nelect = 0  


        #get DFT or GW eigenvalues
        if isinstance(energies_db,(YamboSaveDB,YamboElectronsDB)):
            energies = energies_db.eigenvalues[0,self.lattice.kpoints_indexes] #spin-up
        elif isinstance(energies_db,YamboQPDB):   # Check this works !!!!
            energies = energies_db.expand_eigenvalues(self.lattice)
            print(energies)
        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))
        
        if Restart == False:

           rho_q = self.calculate_rho_finiteq(kindx, eigenvec_q, eigenval_q, Nexcitons)
           omega_q    = self.calculate_omega_finiteq(energies,energies_db,kindx, eigenvec_q, eigenval_q, Nexcitons)

        if Restart == True:

           rho_q = np.load('rho_q.npy')
           omega_q = np.load('omega_vkl_q.npy')

        exit()

        ibz_nkpoints = max(lattice.kpoints_indexes)+1

        #map from bz -> ibz:
        ibz_rho_q     = np.zeros([ibz_nkpoints,self.nvbands,Nexcitons,self.nqpoints])
        ibz_omega_q   = np.zeros([ibz_nkpoints,self.nvbands,Nexcitons,self.nqpoints])
        ibz_kpoints   = lat.red_kpoints

        rho_max = np.zeros([self.nkpoints, self.nvbands, Nexcitons, self.nqpoints])
        omega_max = np.zeros([self.nkpoints, self.nvbands, Nexcitons, self.nqpoints])
        
        for iq in range(self.nqpoints):

            for idx_bz,idx_ibz in enumerate(lattice.kpoints_indexes):

                if idx_ibz == 0:
                   ibz_rho_q[idx_ibz,:,:,iq]   = rho_q[idx_bz,:,:,iq] 
                   ibz_omega_q[idx_ibz,:,:,iq] = omega_q[idx_bz,:,:,iq] 


                if idx_ibz != 0:

                   if idx_ibz != lattice.kpoints_indexes[idx_bz-1]:

                      ibz_rho_q[idx_ibz,:,:,iq]   = rho_q[idx_bz,:,:,iq] 
                      ibz_omega_q[idx_ibz,:,:,iq] = omega_q[idx_bz,:,:,iq] 
                      rho_max[idx_ibz,:,:,iq]   = rho_q[idx_bz,:,:,iq] 
                      omega_max[idx_ibz,:,:,iq] = omega_q[idx_bz,:,:,iq] 


                if idx_ibz == lattice.kpoints_indexes[idx_bz-1]:

                      for i_exc in range(Nexcitons):

                          for v in range(self.nvbands):

                              rho_max[idx_bz,v,i_exc,iq] = max(rho_q[idx_bz,v,i_exc,iq],rho_max[idx_bz - 1,v,i_exc,iq])
                              ibz_rho_q[idx_ibz,v,i_exc,iq]   = rho_max[idx_bz,v,i_exc,iq] 

                              omega_max[idx_bz,v,i_exc,iq] = max(omega_q[idx_bz,v,i_exc,iq],omega_max[idx_bz - 1,v,i_exc,iq])
                              ibz_omega_q[idx_ibz,v,i_exc,iq]   = omega_max[idx_bz,v,i_exc,iq] 

        print('Map from BZ to IBZ completed')

        #get DFT or GW eigenvalues
        if isinstance(energies_db,(YamboSaveDB,YamboElectronsDB)):
            ibz_energies = energies_db.eigenvalues[0,:,self.start_band:self.mband] #spin-up
        elif isinstance(energies_db,YamboQPDB):   # Check this works !!!!
            ibz_energies = energies_db.eigenvalues_qp
        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))

        # set k-path
        kpoints_path = path.get_klist()[:,:3]
        distances = calculate_distances(kpoints_path)
        nkpoints_path = kpoints_path.shape[0]
        na = np.newaxis

        rho_q_path   = np.zeros([1, nkpoints_path, self.nvbands, Nexcitons, self.nqpoints])
        omega_q_path = np.zeros([1, nkpoints_path, self.nvbands, Nexcitons, self.nqpoints])
        
        fullbz_nkpoints = len(lattice.red_kpoints)

        # I(k,omega_band)
        Intensity_q = np.zeros([n_omegas,nkpoints_path]) 
        Intensity_full_BZ = np.zeros([n_omegas,fullbz_nkpoints]) 
        
        sigma_omega = Time/300
        sigma_momentum = Time/50
        label_string_1 = "Time = "
        label_string_2 = "Sigma_w = "
        label_string_3 = "Sigma_q = "
        
        print(f"{label_string_1} {Time} | {label_string_2} {sigma_omega} | {label_string_3} {sigma_momentum}")
       
        Gaussian = self.Gauss_dist(lat, sigma_omega, sigma_momentum, eigenval_q, iq_fixed, Time, Pump_energy, Nexcitons)
        Boltz = self.Boltz_dist(Time, eigenvec_q, eigenval_q, Nexcitons)
        GBdistribution = self.GBdist(Time, Gaussian, Boltz, Nexcitons)

        for i_exc in range(Nexcitons):

            for iq in range(self.nqpoints):

                # interpolate rho along the k-path
                skw_rho_q   = SkwInterpolator(lpratio,ibz_kpoints,ibz_rho_q[na,:,:,i_exc,iq],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
                rho_q_path[0,:,:,i_exc,iq] = skw_rho_q.interp_kpts(kpoints_path).eigens

                # interpolate omega along the k-path
                skw_omega_q = SkwInterpolator(lpratio,ibz_kpoints,ibz_omega_q[na,:,:,i_exc,iq],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
                omega_q_path[0,:,:,i_exc,iq] = skw_omega_q.interp_kpts(kpoints_path).eigens

      
        rho_q_path[rho_q_path < 0] = 0

        # interpolate energies

        skw_energie = SkwInterpolator(lpratio,ibz_kpoints,ibz_energies[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        energies_path = skw_energie.interp_kpts(kpoints_path).eigens

        top_valence_band = np.max(energies_path[0,:,0:self.nvbands])

        energies_path = energies_path - top_valence_band

        omega_q_path = omega_q_path - top_valence_band

        omega_q = omega_q - top_valence_band

        import matplotlib.pyplot as plt
        
        ############################################
        ##### tr-ARPES interpolated along path #####
        ############################################

        step_tr_arpes_along_path = n_omegas // 20  # 5% of total_loops

        start_time = time.time()

        print("")
        print("")
        print("tr-ARPES calculation along path")
        print("")
        print("Progress: 0%, Estimated time remaining: Calculating...")

        for i_o in range(n_omegas):

            for i_exc in range(Nexcitons):

                for iq in range(self.nqpoints):

                    for i_k in range(nkpoints_path):

                        for i_v in range(self.nvbands): 

                            delta_q = 1.0/( omega_band[i_o] - omega_q_path[0, i_k, i_v, i_exc, iq] + Im*omega_width )

                            Intensity_q[i_o,i_k] += 2*np.pi*GBdistribution[iq,i_exc]*rho_q_path[0, i_k, i_v, i_exc, iq]*(-delta_q.imag)

            if (i_o + 1) % step_tr_arpes_along_path == 0 or i_o == n_omegas - 1:
               elapsed_time = time.time() - start_time
               progress = ((i_o + 1) / n_omegas) * 100
               estimated_total_time = (elapsed_time / (i_o + 1)) * n_omegas
               estimated_time_remaining = estimated_total_time - elapsed_time

               # Convert times to hours, minutes, and seconds
               elapsed_hours, elapsed_minutes, elapsed_seconds = self.seconds_to_hms(elapsed_time)
               remaining_hours, remaining_minutes, remaining_seconds = self.seconds_to_hms(estimated_time_remaining)

               print(f"Progress: {progress:.1f}%, Estimated time remaining: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s\n")

        # Final time
        end_time = time.time()
        total_time = end_time - start_time
        total_hours, total_minutes, total_seconds = self.seconds_to_hms(total_time)

        print(f"Total time taken: {total_hours}h {total_minutes}m {total_seconds}s\n")

        ############################################
        ##### tr-ARPES interpolated along path #####
        ############################################

        ###############################################
        ##### tr-ARPES interpolated along full BZ #####
        ###############################################

        step_tr_arpes_full_bz = n_omegas // 20  # 5% of total_loops

        start_time = time.time()

        print("")
        print("")
        print("tr-ARPES calculation full bz")
        print("")
        print("Progress: 0%, Estimated time remaining: Calculating...")

        for i_o in range(n_omegas):

            for i_exc in range(Nexcitons):

                for iq in range(self.nqpoints):

                    for i_k in range(fullbz_nkpoints):

                        for i_v in range(self.nvbands): 

                            delta_q = 1.0/( omega_band[i_o] - omega_q[i_k, i_v, i_exc, iq] + Im*omega_width )

                            Intensity_full_BZ[i_o,i_k] += 2*np.pi*GBdistribution[iq,i_exc]*rho_q[i_k, i_v, i_exc, iq]*(-delta_q.imag)


            if (i_o + 1) % step_tr_arpes_full_bz == 0 or i_o == n_omegas - 1:
               elapsed_time = time.time() - start_time
               progress = ((i_o + 1) / n_omegas) * 100
               estimated_total_time = (elapsed_time / (i_o + 1)) * n_omegas
               estimated_time_remaining = estimated_total_time - elapsed_time

               # Convert times to hours, minutes, and seconds
               elapsed_hours, elapsed_minutes, elapsed_seconds = self.seconds_to_hms(elapsed_time)
               remaining_hours, remaining_minutes, remaining_seconds = self.seconds_to_hms(estimated_time_remaining)

               print(f"Progress: {progress:.1f}%, Estimated time remaining: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s\n")

        # Final time
        end_time = time.time()
        total_time = end_time - start_time
        total_hours, total_minutes, total_seconds = self.seconds_to_hms(total_time)

        print(f"Total time taken: {total_hours}h {total_minutes}m {total_seconds}s\n")

        ###############################################
        ##### tr-ARPES interpolated along full BZ #####
        ###############################################

        X, Y = np.meshgrid(distances, omega_band)
        import matplotlib.pyplot as plt

        # Plot I(k,w)
        if np.amax(Intensity_q) == 0:
           ax.pcolor(X, Y, Intensity_q, vmin = 0.0, vmax = 1.0, cmap='magma', shading='auto')

        else:
           ax.pcolor(X, Y, Intensity_q, vmin = np.amin(Intensity_q), vmax = np.amax(Intensity_q), cmap=cmap_name, shading = 'auto')

        print(np.amax(Intensity_q))

        # Plot Valence Band Energies
        if isinstance(energies_db,(YamboSaveDB,YamboElectronsDB)):

           for i_b in range(self.nvbands):
               ax.plot(distances,energies_path[0,:,i_b],lw=1.0,color='white')
           for i_b in range(self.ncbands):
               ax.plot(distances,energies_path[0,:,i_b+self.nvbands]+scissor,lw=1.0,color='white')
               #plt.plot(distances,energies_path[0,:,i_b+9],lw=0.5,color='r')

        elif isinstance(energies_db,YamboQPDB):   # Check this works !!!!
           for i_b in range(self.nvbands):
               ax.plot(distances,energies_path[0,:,i_b],lw=1.0,color='white')
           for i_b in range(self.ncbands):
               ax.plot(distances,energies_path[0,:,i_b+self.nvbands],lw=1.0,color='white')

        ax.set_xlim((distances[0],distances[-1]))

        ax.set_ylim((omega_1,omega_2-omega_width))
        
        #plt.axhline(np.max(energies_path[0,:,0:self.nvbands]),c='white')
        for kpoint, klabel, distance in path:
            ax.axvline(distance,c='w')
   
        ax.set_xticks(path.distances)
        ax.set_xticklabels(path.klabels)

        ax.set_title(f"Time = {Time}")


        PE_path = np.zeros(n_omegas)
        PE_full_BZ = np.zeros(n_omegas)

        for i_w in range(n_omegas):
            for i_k in range(nkpoints_path):
                PE_path[i_w] += Intensity_q[i_w,i_k]

        for i_w in range(n_omegas):
            for i_k in range(fullbz_nkpoints):
                PE_full_BZ[i_w] += Intensity_full_BZ[i_w,i_k]

        PE_path = PE_path/np.amax(PE_path)
        PE_full_BZ = PE_full_BZ/np.amax(PE_full_BZ)

        bx.plot(PE_path,omega_band, color = 'darkblue', lw = 1.5, label = "PE - Along path")
        bx.legend(loc = 'lower right', frameon = False)
        bx.set_ylim((omega_1,omega_2))
        bx.set_xlim((0.0,max(PE_path)))

        bx.yaxis.set_ticklabels([]) 
        bx.yaxis.set_ticks([]) 

        ind = (0.0, 0.5, 1.0)
        bx.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ind))
        bx.set_xticklabels(['0.0', '0.5', '1.0'])

        bx.tick_params(top=True, labeltop=True, labelbottom=False)

        cx.plot(PE_full_BZ,omega_band, color = 'crimson', lw = 1.5, label = f"PE - Full BZ")
        cx.legend(loc = 'lower right', frameon = False)
        cx.set_ylim((omega_1,omega_2))
        cx.set_xlim((0.0,max(PE_full_BZ)))

        cx.yaxis.set_ticklabels([]) 
        cx.yaxis.set_ticks([]) 
 
        ind = (0.0, 0.5, 1.0)
        cx.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ind))
        cx.set_xticklabels(['0.0', '0.5', '1.0'])

        cx.tick_params(bottom=True, labelbottom=True)

        return 



    ##########################################
    #  ARPES finite-q #
    ##########################################

    def ExcModulusSquare(self,kindx,ax,Nexcitons,Vband,lat,path,SumoverPath,**kwargs):

        from scipy.interpolate import CubicSpline, pchip
        eigenval_q = np.zeros([Nexcitons, self.nqpoints], dtype = complex) 
        eigenvec_q = np.zeros([len(self.eigenvectors[0]), Nexcitons, self.nqpoints], dtype = complex) 

        for iq in range(self.nqpoints):

            yexc_list = self.from_db_file(self.lattice,filename='ndb.BS_diago_Q1',folder='yambo')[iq]     

            for ik in range(Nexcitons):
                eigenval_q[ik,iq] = yexc_list.eigenvalues[ik]

                for t in range(len(self.eigenvectors[0])):
                    eigenvec_q[t,ik,iq] = yexc_list.eigenvectors[ik,t]



        ExcitonQnumber = np.zeros(self.nqpoints)
        ExcitonIndex = np.zeros(self.nqpoints)
        AbsModArray = np.zeros(self.nqpoints)


        ComputedPath = lat.get_path(path)[1]

        if SumoverPath == 'True':

           for i_exc in range(Nexcitons):

               for iq in range(self.nqpoints):

                   AbsMod= 0.0

                   for t,kvc in enumerate(self.table):

                      k,v,c = kvc[0:3]-1   

                      if any(np.array_equal(k, matrix) for matrix in ComputedPath):
                  
                         print('True', k)
 
                         if v == self.unique_vbands[-Vband]:
                   
                            AbsMod += abs2(eigenvec_q[t,i_exc,iq])

                      else:

                         print('False', k)
                   
                   AbsModArray[iq] = AbsMod
                
                   ExcitonQnumber[iq] = np.round( np.sqrt( abs2(lat.ibz_kpoints[iq][0]) + abs2(lat.ibz_kpoints[iq][1]) + abs2(lat.ibz_kpoints[iq][2]) )  ,6)
                   ExcitonIndex[iq] = eigenval_q.real[i_exc,iq]
               '''
               f1 = pchip(ExcitonQnumber, ExcitonIndex)
               f2 = pchip(ExcitonQnumber, AbsModArray)

               x_interp = np.linspace(0, self.nqpoints-1, 1000)

               y1_interp = f1(x_interp)
               y2_interp = f2(x_interp)
 
               ax.plot(x_interp,y1_interp, color = 'black', zorder = 1)
               '''           
               #ax.plot(ExcitonQnumber,ExcitonIndex,color = 'black')
               ax.scatter(ExcitonQnumber,ExcitonIndex, s = 300*AbsModArray, c = 300*AbsModArray, cmap = 'viridis', vmin = 0.0, vmax = 300.0, zorder = 2, edgecolor = 'black')

          
        if SumoverPath == 'False':

           PathinKvalues = lat.car_kpoints[ComputedPath]

           SelectedPath = np.zeros(len(PathinKvalues))

           for i in range(len(ComputedPath)):

              SelectedPath[i] = np.round( np.sqrt( abs2(PathinKvalues[i][0]) + abs2(PathinKvalues[i][1]) + abs2(PathinKvalues[i][2]) )  ,6)

           for i_exc in range(Nexcitons):

               for iq in range(self.nqpoints):

                   AbsMod= 0.0

                   for t,kvc in enumerate(self.table):

                      k,v,c = kvc[0:3]-1   

                      kpoint_on_path = np.round( np.sqrt( abs2(lat.car_kpoints[k][0]) + abs2(lat.car_kpoints[k][1]) + abs2(lat.car_kpoints[k][2]) ),6)

                      if any(np.array_equal(kpoint_on_path, matrix) for matrix in SelectedPath):
                  
                         print('True', kpoint_on_path)
 
                         if v == self.unique_vbands[-Vband]:
                   
                            AbsMod += abs2(eigenvec_q[t,i_exc,iq])

                      else:

                         print('False', kpoint_on_path)
                   
                   AbsModArray[iq] = AbsMod
                
                   ExcitonQnumber[iq] = iq
                   ExcitonIndex[iq] = eigenval_q.real[i_exc,iq]

               f1 = pchip(ExcitonQnumber, ExcitonIndex)
               f2 = pchip(ExcitonQnumber, AbsModArray)

               x_interp = np.linspace(0, self.nqpoints-1, 1000)

               y1_interp = f1(x_interp)
               y2_interp = f2(x_interp)

               ax.plot(x_interp,y1_interp, color = 'black', zorder = 1)
            
               #ax.plot(ExcitonQnumber,ExcitonIndex,color = 'black')
               ax.scatter(ExcitonQnumber,ExcitonIndex, s = 300*AbsModArray, c = 300*AbsModArray, cmap = 'viridis', vmin = 0.0, vmax = 300.0, zorder = 2, edgecolor = 'black')

        ax.set_ylabel('Exciton energies')
        ax.set_xlabel('$|Q|^2$')

        ax.set_xlim(0 - 0.05,(np.round( np.sqrt( abs2(lat.ibz_kpoints[-1][0]) + abs2(lat.ibz_kpoints[-1][1]) + abs2(lat.ibz_kpoints[-1][2]) )  ,6)) + 0.05)


        return

    ##########################################
    #          Exciton Dispersion            #
    ##########################################


    def ExcitonDispersion(self,ax,lat,path,eigenval_q,eigenvec_q,Nexcitons):

        from abipy.core.skw import SkwInterpolator
        # Lattice and Symmetry Variables
        lattice = self.lattice
        cell = (lattice.lat, lattice.red_atomic_positions, lattice.atomic_numbers)
        symrel = [sym for sym,trev in zip(lattice.sym_rec_red,lattice.time_rev_list) if trev==False ]
        time_rev = True
        nelect = 0  
        lpratio=10
        verbose=True
        fermie = 0

        kpoints_path = path.get_klist()[:,:3]
        distances = calculate_distances(kpoints_path)
        nkpoints_path = kpoints_path.shape[0]

        na = np.newaxis

        for i_exc in range(Nexcitons):

            for iq in range(self.nqpoints):

                # interpolate rho along the k-path
                skw_ExcitonDispersion   = SkwInterpolator(lpratio,lat.ibz_kpoints,eigenval_q[na,:,:].real,fermie,nelect,cell,symrel,time_rev,verbose=verbose)
                ExcitonDispersion = skw_ExcitonDispersion.interp_kpts(kpoints_path).eigens


        for i_exc in range(Nexcitons):
            ax.plot(distances,ExcitonDispersion[0,:,i_exc], color = 'black', lw = 1.5)

        ax.set_xlim((distances[0],distances[-1]))
       
        for kpoint, klabel, distance in path:
            ax.axvline(distance,c='black', alpha = 0.3, linestyle = 'dashed')
   
        ax.set_xticks(path.distances)
        ax.set_xticklabels(path.klabels)

        return

    def tr_ExcitonDispersion(self, ax, save, path, omega_1, omega_2, omega_step, sigma, eigenval_q, eigenvec_q, Time, Pump_energy, iq_fixed, Nexcitons, size):

        from abipy.core.skw import SkwInterpolator
        # Lattice and Symmetry Variables
        lattice = self.lattice
        cell = (lattice.lat, lattice.red_atomic_positions, lattice.atomic_numbers)
        symrel = [sym for sym,trev in zip(lattice.sym_rec_red,lattice.time_rev_list) if trev==False ]
        time_rev = True
        nelect = 0  
        lpratio=10
        verbose=True
        fermie = 0

        ibz_kpoints = save.red_kpoints

        sigma_omega = Time/300
        sigma_momentum = Time/30

        kpoints_path = path.get_klist()[:,:3]
        distances = calculate_distances(kpoints_path)
        nkpoints_path = kpoints_path.shape[0]

        na = np.newaxis

        Gaussian = self.Gauss_dist(save, sigma_omega, sigma_momentum, eigenval_q, iq_fixed, Time, Pump_energy, Nexcitons)
        Boltz = self.Boltz_dist(Time, eigenvec_q, eigenval_q, Nexcitons)
        GBdistribution = self.GBdist(Time, Gaussian, Boltz, Nexcitons)


        for i_exc in range(Nexcitons):

            for iq in range(self.nqpoints):

                # interpolate rho along the k-path
                skw_ExcitonDispersion   = SkwInterpolator(lpratio,ibz_kpoints,eigenval_q[na,:,:].real,fermie,nelect,cell,symrel,time_rev,verbose=verbose)
                ExcitonDispersion = skw_ExcitonDispersion.interp_kpts(kpoints_path).eigens

                skw_Exciton_tr_weights = SkwInterpolator(lpratio,ibz_kpoints,GBdistribution[na,:,:].real,fermie,nelect,cell,symrel,time_rev,verbose=verbose)
                Exciton_tr_weights = skw_Exciton_tr_weights.interp_kpts(kpoints_path).eigens


        for i_exc in range(Nexcitons):
            ax.plot(distances,ExcitonDispersion[0,:,i_exc], color = 'black', lw = 1.5)
            dy = Exciton_tr_weights[0,:,i_exc]*size
            ax.fill_between(distances,ExcitonDispersion[0,:,i_exc]+dy,ExcitonDispersion[0,:,i_exc]-dy,alpha=0.75,color='red',linewidth=0)

        ax.set_xlim((distances[0],distances[-1]))
       
        for kpoint, klabel, distance in path:
            ax.axvline(distance,c='black', alpha = 0.3, linestyle = 'dashed')
   
        ax.set_xticks(path.distances)
        ax.set_xticklabels(path.klabels)

        return


    ##########################################
    #          Exciton Dispersion            #
    ##########################################

    ##########################################
    #  SPIN DEPENDENT PART UNDER DEVELOPMENT #
    ##########################################

    def exciton_bs_spin_pol(self,energies,path,excitons=(0,),debug=False):
        """
        Calculate exciton band-structure
        This function should contains the case of non-polarized calculations.
        Now is a first version
            
            Arguments:
            energies -> can be an instance of YamboSaveDB or YamboQBDB
            path     -> path in reduced coordinates in which to plot the band structure
            exciton  -> exciton index to plot
            spin     -> ??
        """
        if self.eigenvectors is None:
            raise ValueError('This database does not contain Excitonic states,'
                              'please re-run the yambo BSE calculation with the WRbsWF option in the input file.')
        if isinstance(excitons, int):
            excitons = (excitons,)
        #get full kmesh
        kpoints = self.lattice.red_kpoints
        path = np.array(path)

        rep = list(range(-1,2))
        kpoints_rep, kpoints_idx_rep = replicate_red_kmesh(kpoints,repx=rep,repy=rep,repz=rep)
        band_indexes = get_path(kpoints_rep,path)
        band_kpoints = kpoints_rep[band_indexes] 
        band_indexes = kpoints_idx_rep[band_indexes]

        if debug:
            for i,k in zip(band_indexes,band_kpoints):
                x,y,z = k
                plt.text(x,y,i) 
            plt.scatter(kpoints_rep[:,0],kpoints_rep[:,1])
            plt.plot(path[:,0],path[:,1],c='r')
            plt.scatter(band_kpoints[:,0],band_kpoints[:,1])
            plt.show()
            exit()

        #get eigenvalues along the path
        if isinstance(energies,(YamboSaveDB,YamboElectronsDB)):
            #expand eigenvalues to the full brillouin zone
            energies_up = energies.eigenvalues[0,self.lattice.kpoints_indexes]
            energies_dw = energies.eigenvalues[1,self.lattice.kpoints_indexes]
            
        elif isinstance(energies,YamboQPDB):
            #expand the quasiparticle energies to the bull brillouin zone
            # Check this is OK
            # To-do
            print('todo')
            print(energies.eigenvalues_qp[self.lattice.kpoints_indexes].shape)
            pad_energies_up = energies.eigenvalues_qp[self.lattice.kpoints_indexes,:,0]
            pad_energies_dw = energies.eigenvalues_qp[self.lattice.kpoints_indexes,:,1]
            print(energies.max_band)
            min_band = energies.min_band       # check this
            print(pad_energies_up.shape)
            print(pad_energies_dw.shape)
            nkpoints, nbands = pad_energies_up.shape
            #exit()
            #pad_energies = energies.eigenvalues_qp[self.lattice.kpoints_indexes]
            #min_band = energies.min_band       # check this
            #nkpoints, nbands = pad_energies.shape
            energies_up = np.zeros([nkpoints,energies.max_band])
            energies_dw = np.zeros([nkpoints,energies.max_band])
            energies_up[:,min_band-1:] = pad_energies_up 
            energies_dw[:,min_band-1:] = pad_energies_dw
        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))

        energies_up, energies_dw = energies_up[band_indexes],energies_dw[band_indexes]

        weights_up, weights_dw = self.get_exciton_weights_spin_pol(excitons)      
        weights_up, weights_dw = weights_up[band_indexes], weights_dw[band_indexes]

        #make top valence band to be zero
        fermi_level = max([ max(energies_up[:,max(self.unique_vbands)]), max(energies_dw[:,max(self.unique_vbands)] ) ])
        energies_up -= fermi_level  
        energies_dw -= fermi_level  
        
        return np.array(band_kpoints), energies_up, energies_dw, weights_up, weights_dw

    def get_exciton_bs_spin_pol(self,energies_db,path,excitons,size_up=1,size_dw=1,space='bands',f=None,debug=False):
        """
        Get a YambopyBandstructure object with the exciton band-structure SPIN POLARIZED
        
            Arguments:
            ax          -> axis extance of matplotlib to add the plot to
            lattice     -> Lattice database
            energies_db -> Energies database, can be either a SaveDB or QPDB
            path        -> Path in the brillouin zone
        """
        from qepy.lattice import Path
        if not isinstance(path,Path): 
            raise ValueError('Path argument must be a instance of Path. Got %s instead'%type(path))
        if space == 'bands':
            if self.spin_pol=='pol':
               bands_kpoints, energies_up, energies_dw, weights_up, weights_dw = self.exciton_bs_spin_pol(energies_db, path.kpoints, excitons, debug)
               nkpoints = len(bands_kpoints)
               plot_energies_up = energies_up[:,self.start_band:self.mband]
               plot_energies_dw = energies_dw[:,self.start_band:self.mband]
               plot_weights_up  = weights_up[:,self.start_band:self.mband]
               plot_weights_dw  = weights_dw[:,self.start_band:self.mband]
        #    elif spin_pol=='pol':
               
        else:
            raise NotImplementedError('TODO')
            eh_size = len(self.unique_vbands)*len(self.unique_cbands)
            nkpoints = len(bands_kpoints)
            plot_energies = np.zeros([nkpoints,eh_size])
            plot_weights = np.zeros([nkpoints,eh_size])
            for eh,(v,c) in enumerate(product(self.unique_vbands,self.unique_cbands)):
                plot_energies[:,eh] = energies[:,c]-energies[:,v]
                plot_weights[:,eh] = weights[:,c] 

        
        if f: plot_weights_up, plot_weights_dw = f(plot_weights_up), f(plot_weights_dw)
        size_plot_up = 100.0 # 1.0/np.max(plot_weights_up)*100.0
        size_plot_dw = 100.0 # 1.0/np.max(plot_weights_dw)*100.0
        ybs_up = YambopyBandStructure(plot_energies_up, bands_kpoints, weights=plot_weights_up, kpath=path, size=size_plot_up)
        ybs_dw = YambopyBandStructure(plot_energies_dw, bands_kpoints, weights=plot_weights_dw, kpath=path, size=size_plot_dw)
        
        #from numpy import arange
        #x = arange(nkpoints)
        #import matplotlib.pyplot as plt
        #for ib1 in range(17):
        #    plt.plot(x,plot_energies_up[:,ib1],'r--')
        #    plt.scatter(x,plot_energies_up[:,ib1],s=plot_weights_up[:,ib1]*size_up*1000,c='red')
        #for ib1 in range(17):
        #    plt.plot(x,plot_energies_dw[:,ib1],'b--')
        #    plt.scatter(x,plot_energies_dw[:,ib1],s=plot_weights_dw[:,ib1]*size_dw*1000,c='blue')
        #plt.show()
        #exit()

        return ybs_up, ybs_dw

    def get_exciton_weights_spin_pol(self,excitons):
    
        """get weight of state in each band for spin-polarized case"""
        if self.spin_pol == 'no':
           print('This function only works for spin-polarized calculations without spin-orbit')
           exit()

        self.unique_vbands_flip = np.unique(self.table[:,1]-1)
        self.unique_cbands_flip = np.unique(self.table[:,2]-1)

        self.mband_flip      = max(self.unique_cbands_flip) + 1
        self.start_band_flip = min(self.unique_vbands_flip)

        #self.mband_up = max(self.unique_cbands_up) + 1
        #self.mband_dw = max(self.unique_cbands_dw) + 1
        #self.start_band_up = min(self.unique_vbands_up)
        #self.start_band_dw = min(self.unique_vbands_dw)

        weights_flip = np.zeros([self.nkpoints,self.mband_flip])
        #weights_dw = np.zeros([self.nkpoints,self.mband_dw])
        
        for exciton in excitons:
            #get the eigenstate
            eivec = self.eigenvectors[exciton-1]
            #add weights
            sum_weights = 0
            for t,kcv in enumerate(self.table):
                k,c,v,c_s,v_s = kcv-1   # We substract 1 to be consistent with python numbering of arrays
                this_weight = abs2(eivec[t])
                #if c_s == 0 and v_s == 0:
                #   weights_up[k,c] += this_weight
                #   weights_up[k,v] += this_weight
                #elif c_s == 1 and v_s == 1: 
                #   weights_dw[k,c] += this_weight
                weights_flip[k,v] += this_weight
                sum_weights += this_weight
            if abs(sum_weights - 1) > 1e-3: raise ValueError('Excitonic weights does not sum to 1 but to %lf.'%sum_weights)
 
        return weights_flip

    def interpolate_spin_pol(self,energies,path,excitons,lpratio=5,f=None,size=1.0,verbose=True,**kwargs):
        """ Interpolate exciton bandstructure using SKW interpolation from
        Abipy and SPIN-POLARIZED CALCULATIONS
        Cuidado con el error in kpoints_indexes 
        """
        from abipy.core.skw import SkwInterpolator

        if verbose:
            print("This interpolation is provided by the SKW interpolator implemented in Abipy")

        lattice = self.lattice
        cell = (lattice.lat, lattice.red_atomic_positions, lattice.atomic_numbers)
        nelect = 0
        # Here there is something strange...

        fermie = kwargs.pop('fermie',0)
        ##
        symrel = [sym for sym,trev in zip(lattice.sym_rec_red,lattice.time_rev_list) if trev==False ]
        time_rev = True
 
        weights_flip = self.get_exciton_weights_spin_pol(excitons)

        #weights_flip = weights_flip[:,self.start_band_flip:self.mband_flip]
        #weights_dw = weights_dw[:,self.start_band_dw:self.mband_dw]

        if f: weights_flip = f(weights_flip)

        size *= 1.0/np.max(weights_flip)

        ibz_nkpoints = max(lattice.kpoints_indexes)+1
        kpoints = lattice.red_kpoints

        #map from bz -> ibz:
        # bug here? it is self.mband, but why?
        ibz_weights_flip = np.zeros([ibz_nkpoints,self.mband_flip-self.start_band_flip]) 
        
        ibz_kpoints = np.zeros([ibz_nkpoints,3])
        print(self.mband_flip,self.start_band_flip)
        print(ibz_weights_flip.shape)
        print(weights_flip.shape)
        print(lattice.kpoints_indexes)
        print('just before error')
        k1,k2,k3 = energies.expand_kpts()
        kpoints_indexes = k2
        # Kpoints indexes must be read from a SAVEDB Class
        for idx_bz,idx_ibz in enumerate(kpoints_indexes):
        #    print(idx_bz,idx_ibz)
        #    print(weights_flip[idx_bz,:])
            ibz_weights_flip[idx_ibz,:] = weights_flip[idx_bz,:]
            ibz_kpoints[idx_ibz] = lattice.red_kpoints[idx_bz]

        print('read eigenvalues')
        print('meter autovalores en tres arrays')
        print(energies.eigenvalues.shape)

        print(self.unique_vbands_flip)
        print(self.unique_cbands_flip)
        print(self.mband_flip)
        print(self.start_band_flip)
        print(ibz_weights_flip.shape)
        ibz_energies = np.zeros([])
        exit()

        #get eigenvalues along the path
        # DFT values from SAVE
        if isinstance(energies,(YamboSaveDB,YamboElectronsDB)):
            ibz_energies_up = energies.eigenvalues[0,:,self.start_band:self.mband] # spin-up channel
            ibz_energies_dw = energies.eigenvalues[1,:,self.start_band:self.mband] # spin-dw channel
            ibz_kpoints_qp  = ibz_kpoints
        # GW values from ndb.QP
        elif isinstance(energies,YamboQPDB):
            ibz_nkpoints_gw=len(energies.kpoints_iku)
            if not ibz_nkpoints == ibz_nkpoints_gw :
               print('GW and BSE k-grid are differents!')
               kpoints_gw_iku = energies.kpoints_iku
               kpoints_gw_car = np.array([ k/lattice.alat for k in kpoints_gw_iku ])
               kpoints_gw_red = car_red( kpoints_gw_car,lattice.rlat)
               ibz_kpoints_qp = kpoints_gw_red
            else:
               ibz_kpoints_qp = ibz_kpoints
            pad_energies_up = energies.eigenvalues_qp[:,:,0]
            pad_energies_dw = energies.eigenvalues_qp[:,:,1]
            min_band = energies.min_band
            nkpoints, nbands = pad_energies_up.shape

            ibz_energies_up = pad_energies_up 
            ibz_energies_dw = pad_energies_dw
            #print('ibz',ibz_energies_up.shape)
        else:
            raise ValueError("Energies argument must be an instance of YamboSaveDB,"
                             "YamboElectronsDB or YamboQPDB. Got %s"%(type(energies)))

        #interpolate energies
        na = np.newaxis

        skw_flip = SkwInterpolator(lpratio,ibz_kpoints_qp,ibz_energies_up[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        skw_dw = SkwInterpolator(lpratio,ibz_kpoints_qp,ibz_energies_dw[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        energies_up = skw_up.interp_kpts(kpoints_path).eigens
        energies_dw = skw_dw.interp_kpts(kpoints_path).eigens
     
        #interpolate weights
        na = np.newaxis
        skw_up = SkwInterpolator(lpratio,ibz_kpoints,ibz_weights_up[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        skw_dw = SkwInterpolator(lpratio,ibz_kpoints,ibz_weights_dw[na,:,:],fermie,nelect,cell,symrel,time_rev,verbose=verbose)
        kpoints_path = path.get_klist()[:,:3]
        exc_weights_up = skw_up.interp_kpts(kpoints_path).eigens
        exc_weights_dw = skw_dw.interp_kpts(kpoints_path).eigens

        # Find and set the up-dw Fermi energy to zero
        self.nvbands_up = len(self.unique_vbands_up)
        self.nvbands_dw = len(self.unique_vbands_dw)
        fermi_up_dw = max([max(energies_up[0][:,self.nvbands_up-1]), max(energies_dw[0][:,self.nvbands_dw-1])])

        #create band-structure object
        exc_bands_up = YambopyBandStructure(energies_up[0],kpoints_path,kpath=path,fermie=fermi_up_dw,weights=exc_weights_up[0],size=size_up,**kwargs)
        exc_bands_dw = YambopyBandStructure(energies_dw[0],kpoints_path,kpath=path,fermie=fermi_up_dw,weights=exc_weights_dw[0],size=size_dw,**kwargs)

        return exc_bands_up, exc_bands_dw

    ##############################################
    #  END SPIN DEPENDENT PART UNDER DEVELOPMENT #
    ##############################################



    def seconds_to_hms(self,seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(hours), int(minutes), int(seconds)












    def get_string(self,mark="="):
        lines = []; app = lines.append
        app( marquee(self.__class__.__name__,mark=mark) )
        app( "BSE solved at Q:            %s"%self.Qpt )
        app( "number of excitons:         %d"%self.nexcitons )
        if self.table is not None: 
            app( "number of transitions:      %d"%self.ntransitions )
            app( "number of kpoints:          %d"%self.nkpoints  )
            app( "number of valence bands:    %d"%self.nvbands )
            app( "number of conduction bands: %d"%self.ncbands )
        return '\n'.join(lines)
    
    def __str__(self):
        return self.get_string()
