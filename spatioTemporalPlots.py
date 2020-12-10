# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:01:05 2020

@author: dolor
"""

import MC_snow as snow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
    
muS = medium_params['muS']
muA = medium_params['muA']
g = medium_params['g']
c_snow = medium_params['c_snow']

# %% 

folder = './spatiotemporalTest/'
filename = 'grain_radius_{}_density_{}_wavelength_{}.npy'.format(r_snow, density_snow, wavelength)

with open(folder + 'data_'+filename, 'rb') as f:
    data = np.load(f)
    
with open(folder + 'counters_'+filename, 'rb') as f:
    counters = np.load(f)  
    
print('Data loaded for spatiotemporal test.') 

s = np.sqrt(data[:, 2]**2 + data[:, 3]**2)

plt.hist2d(s, data[:, 1]/c_snow, weights = 1/s, bins = (100, 100), norm = colors.LogNorm()); plt.colorbar()

# %% 

tmin = 0.
tmax = 1E-6
num_t = 100

smin = 0.
smax = 1.5
num_s = 100

plt.hist2d(100*s, 1E9*data[:, 1]/c_snow, weights = 1/s, bins = (num_t, num_s), \
           norm = colors.LogNorm(), range=[[100*smin, 100*smax],[1E9*tmin, 1E9*tmax]]); 
plt.gca().invert_yaxis(); plt.colorbar(); 
plt.xlabel('Distance from laser spot (cm)');
plt.ylabel('Time (ns)');
plt.title('Photon Counts vs. time, distance from laser spot')
plt.show(); 

# H, _ = np.hist2d(s, data[:, 1]/c_snow, weights=1/s, bins = (num_t, num_s), \
#                  range=[[smin, smax],[tmin, tmax]])
    
# tbins = np.linspace(tmin, tmax, num_t)
# sbins = np.linspace(smin, smax, num_s)

s_o = 0.3 # Distance between laser spot and observed spot on snowpack surface
w_o = 0.01 # 
num_bins = 500
max_tt = 1E-6

p_true = np.zeros(4)
p_fit = np.zeros(4)

#s_o = s[(s>(observe_radius-observe_width)) and (s<(observe_radius+observe_width))]
t_o = data[(s>(s_o-w_o)) & (s<(s_o+w_o)), 1]/c_snow

#plt.hist(t_o, bins=500, log=True)

hist, _ = np.histogram(t_o, bins=num_bins, range=(0, max_tt))

bins = np.linspace(0, max_tt, num_bins)

#plt.plot(bins, hist); plt.yscale('log')

z0 = 1/(muS*(1-g))
D = 1/(3*(muA + (1-g)*muS))

p_true[0] = np.log(z0/(4*np.pi*D*c_snow)**1.5)
p_true[1] = -muA*c_snow
p_true[2] = -s_o**2 / (4*D*c_snow)
p_true[3] = -2.5

diff_curve_true = p_true[0] + p_true[1]*bins + p_true[2]/bins + p_true[3]*np.log(bins)
diffusion_offset = np.sum(hist)/np.nansum(np.exp(diff_curve_true))

p_fit, V_fit = snow.fitLargeFootprintDiffusionCurve(bins, hist)
diff_curve_fit = p_fit[0] + p_fit[1]*bins + p_fit[2]/bins + p_fit[3]*np.log(bins)

print('Snow r = {} um, rho = {} kg/m3, wavelength = {} nm .'.format(r_snow*1E6, density_snow, wavelength*1E9))
print('True diffusion parameters: {}'.format(p_true))
print('Fitted diffusion parameters: {}'.format(p_fit))

# Time-of-flight histograms + Diffusion curves
plt.plot(1E9*bins, 10*np.log10(hist)); 
plt.plot(1E9*bins, 10*diff_curve_true*np.log10(np.e) + \
         10*np.log10(diffusion_offset), linestyle='--')
plt.plot(1E9*bins, 10*diff_curve_fit*np.log10(np.e), linestyle='-.')

plt.ylim(bottom=0)
plt.legend(('Counts', 'Diffusion', 'Fit'))
plt.xlabel('Photon travel-time (ns)')
plt.ylabel('Counts')
plt.title(r'Time-of-flght histogram for Spatial offset = {} cm'.format(s_o*100))
plt.show()

tts = np.linspace(tmin, tmax, num_t)
sss = np.linspace(smin, smax, num_s)

TTS, SSS = np.meshgrid(tts, sss)
p2zs = p_true[0] + p_fit[1]*TTS - (z0**2 + SSS**2)/(4*D*c_snow*TTS) + p_true[3]*np.log(TTS)
plt.pcolor(100*sss, 1E9*tts, np.transpose(np.exp(p2zs)), norm = colors.LogNorm(), vmin=1)
plt.colorbar()
plt.gca().invert_yaxis(); 
plt.xlabel('Distance from laser spot (cm)');
plt.ylabel('Time (ns)');
plt.title('Diffusion Approximation vs. time, distance from laser spot')
plt.show(); 

# %% Inversion

r_lims = [10E-6, 2E-3] # m
num_r = 200

rho_lims = [50, 750] # kg/m3
num_rho = 200

p1s, p2s, rs, rhos = snow.diffusionParameterLookupTable_offset( \
                            wavelength, m_ice, s_o, \
                            r_lims, num_r, rho_lims, num_rho)
    
loss_true = (p1s-p_true[1])**2/(p1s**2) + (p2s-p_true[2])**2/(p2s**2)
plt.pcolor(rhos, 1E6*rs, np.log(loss_true))
plt.colorbar()
plt.title('Log loss, True Diffusion, True r =  {} um,  rho = {} kg/m3'.format(1E6*r_snow, density_snow))
plt.xlabel('Snow density (kg/m3)')
plt.ylabel('Grain size (um)')
plt.show()
amin_true = np.unravel_index(np.argmin(loss_true), loss_true.shape)
print('True diffusion fit: r =  {},  rho = {}'.format(rs[amin_true[0]], rhos[amin_true[1]]))

loss_fit = (p1s-p_fit[1])**2/(p1s**2) + (p2s-p_fit[2])**2/(p2s**2)
plt.pcolor(rhos, 1E6*rs, np.log(loss_fit))
plt.title('Log loss, True r =  {} um,  rho = {} kg/m3'.format(1E6*r_snow, density_snow))
plt.xlabel('Snow density (kg/m3)')
plt.ylabel('Grain size (um)')
amin_fit = np.unravel_index(np.argmin(loss_fit), loss_fit.shape)
plt.colorbar()
plt.show()
print('Estimate from fit: r =  {},  rho = {}'.format(rs[amin_fit[0]], rhos[amin_fit[1]]))






