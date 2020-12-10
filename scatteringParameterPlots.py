# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:12:37 2020

@author: dolor
"""

import sys
sys.path.insert(0,'C:/Users/dolor/PyMieScatt/')
import PyMieScatt as ps
import numpy as np
import matplotlib.pyplot as plt


# %% Table generation function

def scatteringParameterLookupTable(wavelength, m_ice, r_lims, num_r, rho_lims, num_rho):

    c = 299792458 # Speed of light (m/s)
    
    density_ice = 920 # kg / m3 at T = -30 C, P = 1 atm
    density_air = 1.451 # kg / m3 at T = -30 C, P = 1 atm
    
    r = np.linspace(r_lims[0], r_lims[1], num_r)
    rho = np.linspace(rho_lims[0], rho_lims[1], num_rho)
    
    f_snow = (rho - density_air)/(density_ice - density_air)
    
    muSs = np.zeros((num_r, num_rho))
    muAs = np.zeros_like(muSs)
    gs = np.zeros_like(muSs)
    c_snows = np.zeros_like(muSs)
    
    for jj in range(0, num_rho):
        n_snow = 1 + (m_ice.real - 1) * (rho[jj] - density_air)/(density_ice - density_air)
        c_snows[:, jj] = c/n_snow
    
    for ii in range(0, num_r):        
        
        _, Qs, Qa, g, _, _, _ = ps.MieQ(m_ice, wavelength*1E9, 2*r[ii]*1E9)
        gs[ii, :] = g
        
        for jj in range(0, num_rho):
            
            muSs[ii, jj] = 3*(rho[jj] - density_air)*Qs/(4*(density_ice - density_air)*r[ii])
            muAs[ii, jj] = 3*(rho[jj] - density_air)*Qa/(4*(density_ice - density_air)*r[ii])
    
    return muSs, muAs, c_snows, r, rho, f_snow

# %% Generate tables 532 nm

m_ice_532 = 1.3116 + 1.4898e-9j
n532 = m_ice_532.real
m_ice = m_ice_532
wavelength = 532.0E-9

r_lims = [10E-6, 2E-3] # m
num_r = 200

rho_lims = [50, 750] # kg/m3
num_rho = 281

muSs532, muAs532, c_snows532, r, rho, f_snow = scatteringParameterLookupTable(wavelength, m_ice, r_lims, num_r, rho_lims, num_rho)

# %% Generate tables 1064 nm

m_ice_1064 = 1.3004 + 1.9e-6j
n1064 = m_ice_1064.real
m_ice = m_ice_1064
wavelength = 1064.0E-9

muSs1064, muAs1064, c_snows1064, _, _, _ = scatteringParameterLookupTable(wavelength, m_ice, r_lims, num_r, rho_lims, num_rho)

# %% Light speed plot

c = 299792458 # Speed of light (m/s)

plt.plot(f_snow, c*np.ones_like(f_snow), '--y', \
         f_snow, c_snows532[0, :], 'g', \
         f_snow, c_snows1064[0, :], 'r', \
         f_snow, c*np.ones_like(f_snow)/n532, '--g', \
         f_snow, c*np.ones_like(f_snow)/n1064, '--r'); 
plt.title('Speed of light in snow vs. snowpack density'); 
plt.xlabel('Ice volume fraction'); 
plt.ylabel('Mean light speed (m/s)'); 
plt.legend(('Vacuum', '$c_*$ (532 nm)', \
            '$c_*$ (1064 nm)', '$c_{ice}$ (532 nm)', '$c_{ice}$ (1064 nm)'), \
            loc='upper right')






