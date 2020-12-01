# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:10:21 2020

@author: dolor
"""

import sys
sys.path.insert(0,'C:/Users/dolor/PyMieScatt/')
import PyMieScatt as ps
import numpy as np
import scipy.interpolate as interpolate

c = 299792458 # Speed of light (m/s)

m_ice_532 = 1.3116 + 1.4898e-9j
m_ice_1064 = 1.3004 + 1.9e-6j

wavelength = 532.0E-9

r_ice = 200E-6
m_ice = m_ice_532
density_snow = 250 # kg/m3 For moderately settled snow, but still dry

m_soot = 1.8 + 0.5j # not wavelength dependent in WWII 1981
r_soot = 0.1E-6
f_soot = 1E-6 # density_soot/density_snow (1E-6 = 1 ppmw soot)

pfun_bins = 1000

def snowpackScatterProperties(wavelength, m_ice, r_ice, density_snow, 
                                         m_soot, r_soot, f_soot, 
                                         snow_depth=np.inf, mie=False,
                                         pfun_bins=1000,  pfunsmoothing=True):
    
    c = 299792458 # Speed of light (m/s)

    density_ice = 920 # kg / m3 at T = -30 C, P = 1 atm
    density_air = 1.451 # kg / m3 at T = -30 C, P = 1 atm
    
    icegrain_properties = ps.MieQ(m_ice, wavelength*1E9, r_ice*1E9, asDict=True)
    
    #N_snow = 3*(density_snow - density_air)/(4*math.pi*( density_ice - density_air )*r_ice**3)
    
    Qsca_ice = icegrain_properties['Qsca']
    Qabs_ice = icegrain_properties['Qabs']
    g_ice = icegrain_properties['g']
    
    muS_ice = 3*(density_snow - density_air)*Qsca_ice/(4*(density_ice - density_air)*r_ice)
    muA_ice = 3*(density_snow - density_air)*Qabs_ice/(4*(density_ice - density_air)*r_ice)
    
    n_snow = 1 + (m_ice.real - 1) * (density_snow - density_air)/(density_ice - density_air)
    c_snow = c / n_snow
    
    sootgrain_properties = ps.MieQ(m_soot, wavelength*1E9, r_soot*1E9, asDict=True)
    
    Qsca_soot = sootgrain_properties['Qsca']
    Qabs_soot = sootgrain_properties['Qabs']
    g_soot = sootgrain_properties['g']
    
    muS_soot = 3*f_soot*Qsca_soot/(4*r_soot)
    muA_soot = 3*f_soot*Qabs_soot/(4*r_soot)
    
    muS = muS_ice + muS_soot
    muA = muA_ice + muA_soot
    g = (1/muS)*( muS_ice*g_ice + muS_soot*g_soot )
    
    medium_params = {'muS': muS, 'muA': muA, 'g': g, 'c_snow': c_snow}
    if np.isinf(snow_depth) or snow_depth <= 0:
        medium_params['z_bounded'] = False
        medium_params['z_range'] = np.array([0. , np.inf])
    else:
        medium_params['z_bounded'] = True
        medium_params['z_range'] = np.array([0. , snow_depth])
    
    if mie:
    # Disregarding contribution of soot to phase function for now, because it 
    # should be very small.  But if want to be more rigorous, can calculate
    # phase functions for soot and ice grains and then average, weighted by
    # scattering coefficients.  Because soot particles are small vs. wavelength, 
    # likely do not need to use smoothing.  Phase function for single radius 
    # should already be smooth
        if pfunsmoothing:
            # Use sum of phase functions from uniform distribution of grain radii 
            # instead of single phase function.
            # Result is smoother but compute time is longer
            P = miePhaseFunctionSmooth(m_ice, wavelength, r_ice, pfun_bins, .01*r_ice, 20)
        else:
            P = miePhaseFunction(m_ice, wavelength, r_ice, pfun_bins)
            
            
        bin_edges = np.linspace(-1, 1, 1001)
        cdf_P = np.zeros_like(bin_edges)
        cdf_P[1:] = np.cumsum(P)
        inv_cdf_P = interpolate.interp1d(cdf_P, bin_edges)
        
        medium_params['mie'] = True
        medium_params['P'] = P, 
        medium_params['inv_cdf_P'] = inv_cdf_P
    else:
        medium_params['mie'] = False
    return medium_params

def miePhaseFunction(m, wavelength, r, num_bins):

    cosines1 = np.linspace(-1+1/num_bins, 1-1/num_bins, num_bins)
    cosines2 = np.linspace(1.-2./num_bins + 1./num_bins**2, 1.-1./num_bins**2, num_bins)
    P1 = np.zeros_like(cosines1)
    P2 = np.zeros_like(cosines2)

    for ii in range(0, num_bins):
        P1[ii], _, _, _ = ps.MatrixElements(m, wavelength*1E9, r*1E9, cosines1[ii])
        P2[ii], _, _, _ = ps.MatrixElements(m, wavelength*1E9, r*1E9, cosines2[ii])
    
    P1[-1] = np.mean(P2)
    P1 = P1/np.sum(P1)
    return P1

def miePhaseFunctionSmooth(m, wavelength, r, num_bins, dr, num_r):
    
    radii = np.linspace(r-dr, r+dr, num_r)
    
    P = np.zeros(num_bins)
    
    for radius in radii:
        
        print(radius)

        cosines1 = np.linspace(-1+1/num_bins, 1-1/num_bins, num_bins)
        cosines2 = np.linspace(1.-2./num_bins + 1./num_bins**2, 1.-1./num_bins**2, num_bins)
        P1 = np.zeros_like(cosines1)
        P2 = np.zeros_like(cosines2)

        for ii in range(0, num_bins):
            P1[ii], _, _, _ = ps.MatrixElements(m, wavelength*1E9, radius*1E9, cosines1[ii])
            P2[ii], _, _, _ = ps.MatrixElements(m, wavelength*1E9, radius*1E9, cosines2[ii])
    
        P1[-1] = np.mean(P2)
        
        P = P + P1
        P = P/np.sum(P)
        
    return P

# How to sample a phase function CDF
# inv_cdf_P can be the input to propPhotonMieGPU
# bin_edges = np.linspace(-1, 1, 1001)
# cdf_P = np.zeros_like(bin_edges)
# cdf_P[1:] = np.cumsum(P)
# inv_cdf_P = interpolate.interp1d(cdf_P, bin_edges)
# rng = np.random.default_rng()

# rand1 = rng.random()
# mu = inv_cdf_p(rand1)

    

