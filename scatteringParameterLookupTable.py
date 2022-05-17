# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 16:37:37 2022

@author: dolor

"""

import sys
sys.path.insert(0,'C:/Users/dolor/PyMieScatt/')
import PyMieScatt as ps
import numpy as np
import matplotlib.pyplot as plt


#m_ice_532 = 1.3116 + 1.4898e-9j
m_ice_635 = 1.3084 + 1.1300e-8j
m_ice = m_ice_635
wavelength = 635.0E-9 #532.0E-9

r_lims = [10E-6, 2E-3] # m
num_r = 200

rho_lims = [50, 750] # kg/m3
num_rho = 200

def scatteringParameterLookupTable(wavelength, m_ice, r_lims, num_r, rho_lims, num_rho):

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

    
    return muSs, muAs, gs, c_snows,  r, rho
    
muSs, muAs, gs, c_snows,  r, rho = scatteringParameterLookupTable(wavelength, m_ice, r_lims, num_r, rho_lims, num_rho)
    