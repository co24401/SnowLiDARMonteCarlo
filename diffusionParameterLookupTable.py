# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:35:23 2020

@author: dolor
"""

import sys
sys.path.insert(0,'C:/Users/dolor/PyMieScatt/')
import PyMieScatt as ps
import numpy as np
import matplotlib.pyplot as plt


m_ice_532 = 1.3116 + 1.4898e-9j
m_ice = m_ice_532
wavelength = 532.0E-9

r_lims = [10E-6, 2E-3] # m
num_r = 200

rho_lims = [50, 750] # kg/m3
num_rho = 200

def diffusionParameterLookupTable(wavelength, m_ice, r_lims, num_r, rho_lims, num_rho):

    c = 299792458 # Speed of light (m/s)
    
    density_ice = 920 # kg / m3 at T = -30 C, P = 1 atm
    density_air = 1.451 # kg / m3 at T = -30 C, P = 1 atm
    
    r = np.linspace(r_lims[0], r_lims[1], num_r)
    rho = np.linspace(rho_lims[0], rho_lims[1], num_rho)
    
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

    z0s = 1/(muSs*(1-gs))
    Ds = 1/(3*(muAs + (1-gs)*muSs))
    
    a1s = -muAs*c_snows
    a2s = -z0s**2/(4*Ds*c_snows)
    
    return a1s, a2s, r, rho
    
a1s, a2s, rs, rhos = diffusionParameterLookupTable(wavelength, m_ice, r_lims, num_r, rho_lims, num_rho)
 
# Example values for r = 200 um, rho = 250 kg/m3   
a1_true = -3410485.9268168337    
a2_true = -1.214188425961465e-11

loss = (a1s-a1_true)**2/(a1s**2) + (a2s-a2_true)**2/(a2s**2)
plt.imshow(np.log(loss))
amin = np.unravel_index(np.argmin(loss), loss.shape)
print('r =  {},  rho = {}'.format(rs[amin[0]], rhos[amin[1]]))

# If you only have a1
loss_a1 = (a1s-a1_true)**2/(a1s**2)
amin_rho = np.argmin(np.sum(loss_a1, axis=0))
plt.plot(np.log(np.sum(loss_a1, axis=0)))
print('rho = {}'.format(rhos[amin_rho]))
    