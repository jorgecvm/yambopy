import matplotlib.pyplot as plt
from qepy import *
from yambopy import *

# Matplotlib options
fig = plt.figure(figsize=(10,10))
ax  = fig.add_axes( [ 0.15, 0.15, 0.75, 0.75 ])

# Reading RT database
#
# Notice that if you have two valence bands and two conduction band, the first conduction band would be sel_band = 2)

Carriers = YamboRTcarriers(carriers_file = 'ndb.RT_carriers', save_path = 'SAVE/ns.db1', sel_band = 2)

# Plotting carriers
#
# Mode can be 'raw' or 'interpolated
#
# Mode_BZ can be 'simple' or 'repetition'

Carriers.plot_carriers_ax(ax, time = 400, vmin = -2e-3, vmax = 2e-3, cmap = 'PiYG', color_Wigner = 'black', lw_Wigner = '0.75', size  = 30, mode = 'raw', mode_BZ = 'repetition')

plt.show()




