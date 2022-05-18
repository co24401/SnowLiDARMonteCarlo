# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:36:21 2022

@author: dolor
"""

import MC_snow as snow
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# %%

# Value taken from refractiveindex.info, but data on that site was taken from 
# Warren and Brandt 2008
m_ice_640 = 1.3083 + 1.2200e-8j # Basement laser is 640 nm
wavelength = 640.0E-9 #532.0E-9
m_ice = m_ice_640
density_snow = 166.9 # kg/m3 For moderately settled snow, but still dry
r_snow = 100E-6

# Not used but including because will break my code otherwise
m_soot = 1.8 + 0.5j # not wavelength dependent in WWII 1981
r_soot = 0.1E-6
f_soot = 0.#1E-6 # density_soot/density_snow (1E-6 = 1 ppmw soot)

medium_params = snow.snowpackScatterProperties(wavelength, m_ice, r_snow, \
                density_snow, m_soot, r_soot, f_soot, pfunsmoothing=False, mie=False)
    
muS = medium_params['muS']
muA = medium_params['muA']
g = medium_params['g']
c_snow = medium_params['c_snow']

# %%

filename = 'grain_radius_{}_density_{}_wavelength_{}.npy'.format(r_snow, density_snow, wavelength)

with open('data_'+filename, 'wb') as f:
    data = np.load(f)
    
with open('counters_'+filename, 'wb') as f:
    counters = np.load(f)  
    
print('Data loaded from spatiotemporal test.') 

# %%

folder = filename[0:-4]

s = np.sqrt(data[:, 2]**2 + data[:, 3]**2)

tmin = 0.0;
tmax = 400E-9;
tbin = 8e-12;

smin = 0.0;
smax = 1.0;
sbin = 0.00635;

t_edges = np.arange(tmin, tmax, tbin)
s_edges = np.arange(smin, smax, sbin)

radialHistos, s_edges, t_edges = np.histogram2d(s, data[:,1]/c_snow, bins=[s_edges, t_edges])
params = np.array([muS, muA, g, c_snow, tmin, tmax, tbin, smin, smax, sbin])

#os.mkdir(folder)
np.savetxt('./' + folder + '/radialHistos.csv', radialHistos, delimiter=',')
np.savetxt('./' + folder + '/params.csv', params, delimiter=',')