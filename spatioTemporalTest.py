# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:38:26 2020

@author: dolor
"""
import MC_snow as snow
import numpy as np

# %% 

simulation_params = {'device_id': 0, 'batchSize': 1044480*2, 
                                  'nPhotonsRequested': 1044480*40, 
                                  'nPhotonsToRun': 5e8, 
                                  'max_N': 1e6, 
                                  'max_distance_from_det': 1000.0, 
                                  'quantized': True}

source_params = {'pos': np.array([0., 0., 0.]), 
                 'mu': np.array([0., 0., 1.]), 
                 'radius': 0.}

detector_params = {'radius': 5.}

# %% Generate medium

m_ice_532 = 1.3116 + 1.4898e-9j
m_ice_1064 = 1.3004 + 1.9e-6j

wavelength = 532.0E-9
m_ice = m_ice_532
density_snow = 250 # kg/m3 For moderately settled snow, but still dry
r_snow = 200E-6

# Not used but including because will break my code otherwise
m_soot = 1.8 + 0.5j # not wavelength dependent in WWII 1981
r_soot = 0.1E-6
f_soot = 0.#1E-6 # density_soot/density_snow (1E-6 = 1 ppmw soot)

medium_params = snow.snowpackScatterProperties(wavelength, m_ice, r_snow, \
                density_snow, m_soot, r_soot, f_soot, pfunsmoothing=False, mie=False)
    
print('Snowpack properties generated')

# %% Run simulation

data, counters = snow.run_d_MC(simulation_params, medium_params, 
                                      source_params, detector_params)

print('Simulation completed for spatiotemporal test')

folder = './spatiotemporalTest/'
filename = 'grain_radius_{}_density_{}_wavelength_{}.npy'.format(r_snow, density_snow, wavelength)

with open(folder + 'data_'+filename, 'wb') as f:
    np.save(f, data)
    
with open(folder + 'counters_'+filename, 'wb') as f:
    np.save(f, counters)  
    
print('Data saved for spatiotemporal test.') 
    
