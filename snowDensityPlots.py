# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:29:01 2020

@author: dolor

"""

import MC_snow as snow
import numpy as np
import matplotlib.pyplot as plt

# %%

c = 299792458.

m_ice_532 = 1.3116 + 1.4898e-9j
m_ice_1064 = 1.3004 + 1.9e-6j

# Not used but including because will break my code otherwise
m_soot = 1.8 + 0.5j # not wavelength dependent in WWII 1981
r_soot = 0.1E-6
f_soot = 0.#1E-6 # density_soot/density_snow (1E-6 = 1 ppmw soot)

nPhotonsRequested = 1044480*4;

data = np.zeros((3, nPhotonsRequested))
counters = np.zeros((3, 3))
muSs = np.zeros(3)
muAs = np.zeros(3)
gs = np.zeros(3)
c_snows = np.zeros(3)

p_true = np.zeros((3, 4))
p_true[:, 3] = -1.5

p_fit = np.zeros((3, 4))

# %% Load data and look at the histograms to choose bin size/window for plots 
# in the next cell
# Radius tests 532 nm

wavelength = 532.0E-9
m_ice = m_ice_532
snow_densities = [80., 250., 500.]
r_snow = 200E-6

for ii, density_snow in enumerate(snow_densities):
    
    print('Simulation completed for snow density = {}'.format(density_snow))
    
    folder = './snowDensityTests/'
    filename = 'grain_radius_{}_density_{}_wavelength_{}.npy'.format(r_snow, density_snow, wavelength)
    
    with open(folder + 'data_' + filename, 'rb') as f:
        batch_data = np.load(f)
        
    data[ii, :] = batch_data[:, 1]
        
    with open(folder + 'counters_' + filename, 'rb') as f:
        counters[ii, :] = np.load(f)
        
    plt.hist(data[ii, :], bins=500, log='True');
    plt.show()
        
    print('Data loaded for snow r = {}, rho = {}'.format(r_snow, density_snow))
    print('Albedo measured to be {}'.format(counters[ii, 1]/counters[ii, 0]))
    
# %%  Plot histograms, diffusion curves and apply fits
# Radius tests 532 nm

max_travel_dist = 1000. # meters
num_bins = 500

bins = np.linspace(0, max_travel_dist/c, num_bins)

hists = np.zeros((3, num_bins))
diff_curves_true = np.zeros_like(hists)
diffusion_offsets = np.zeros(3)
diff_curves_fit = np.zeros_like(hists)
    
for ii, density_snow in enumerate(snow_densities):
    
    medium_params = snow.snowpackScatterProperties(wavelength, m_ice, r_snow, \
                    density_snow, m_soot, r_soot, f_soot, pfunsmoothing=False, mie=False)

    muSs[ii] = medium_params['muS']
    muAs[ii] = medium_params['muA']
    gs[ii] = medium_params['g']
    c_snows[ii] = medium_params['c_snow']
        
    #bins[ii, :] = np.linspace(0, max_travel_dist./c_snows[ii], num_bins)
    hists[ii, :], _ = np.histogram(data[ii, :]/c_snows[ii], bins=num_bins, \
                                range=(0, max_travel_dist/c))
    z0 = 1/(muSs[ii]*(1-gs[ii]))
    D = 1/(3*(muAs[ii] + (1-gs[ii])*muSs[ii]))
    
    p_true[ii, 0] = np.log(z0/np.sqrt(4*np.pi*D*c_snows[ii]))
    p_true[ii, 1] = -muAs[ii]*c_snows[ii]
    p_true[ii, 2] = -z0**2/(4*D*c_snows[ii])
    diff_curves_true[ii, :] = p_true[ii, 0] + p_true[ii, 1]*bins + p_true[ii, 2]/bins + p_true[ii, 3]*np.log(bins)
    diffusion_offsets[ii] = np.sum(hists[ii, 3:])/np.nansum(np.exp(diff_curves_true[ii, 3:]))
        
    p_fit[ii, :], _ = snow.fitLargeFootprintDiffusionCurve(bins[1:], hists[ii, 1:])
    diff_curves_fit[ii, :] = p_fit[ii, 0] + p_fit[ii, 1]*bins + p_fit[ii, 2]/bins + p_fit[ii, 3]*np.log(bins)
    
    print('Snow r = {} um, rho = {} kg/m3, wavelength = {} nm .'.format(r_snow*1E6, density_snow, wavelength*1E9))
    print('True diffusion parameters: {}'.format(p_true[ii, :]))
    print('Fitted diffusion parameters: {}'.format(p_fit[ii, :]))
    
# Time-of-flight histograms + Diffusion curves
plt.plot(1E9*bins, 10*np.log10(hists[0, :]*counters[0, 1]/counters[0, 0])); 
plt.plot(1E9*bins, 10*np.log10(hists[1, :]*counters[1, 1]/counters[1, 0]));
plt.plot(1E9*bins, 10*np.log10(hists[2, :]*counters[2, 1]/counters[2, 0]));
plt.plot(1E9*bins[1:], 10*diff_curves_true[0, 1:]*np.log10(np.e) + \
         10*np.log10(diffusion_offsets[0]) + 10*np.log10(counters[0, 1]/counters[0, 0]), \
         linestyle='--', color='tab:blue')
plt.plot(1E9*bins[1:], 10*diff_curves_true[1, 1:]*np.log10(np.e) + \
         10*np.log10(diffusion_offsets[1]) + 10*np.log10(counters[1, 1]/counters[1, 0]), \
     linestyle='--', color='tab:orange')
plt.plot(1E9*bins[1:], 10*diff_curves_true[2, 1:]*np.log10(np.e) + \
         + 10*np.log10(diffusion_offsets[2]) + 10*np.log10(counters[1, 1]/counters[1, 0]), \
         linestyle='--', color='tab:green')
plt.ylim(bottom=0)
plt.legend((r'Counts: $\rho_*$ = {}kg/m3'.format(snow_densities[0]), \
            r'Counts: $\rho_*$ = {}kg/m3'.format(snow_densities[1]), \
            r'Counts: $\rho_*$ = {}kg/m3'.format(snow_densities[2]), \
            r'Diffusion: $\rho_*$ = {}kg/m3'.format(snow_densities[0]), \
            r'Diffusion: $\rho_*$ = {}kg/m3'.format(snow_densities[1]), \
            r'Diffusion: $\rho_*$ = {}kg/m3'.format(snow_densities[2])), loc='upper right')
plt.xlabel('Photon travel-time (ns)')
plt.ylabel('Counts')
plt.title(r'Time-of-flght histograms, $r_*$ = {} um, $\lambda$ = {} nm, Various $\rho_*$'.format(density_snow, wavelength*1E9))
plt.show()

# Fitted vs True diffusion curves
plt.plot(1E9*bins[1:], 10*diff_curves_fit[0, 1:]*np.log10(np.e)+10*np.log10(counters[0, 1]/counters[0, 0])); 
plt.plot(1E9*bins[1:], 10*diff_curves_fit[1, 1:]*np.log10(np.e) +10*np.log10(counters[1, 1]/counters[1, 0]));
plt.plot(1E9*bins[1:], 10*diff_curves_fit[2, 1:]*np.log10(np.e) + 10*np.log10(counters[2, 1]/counters[2, 0]));
plt.plot(1E9*bins[1:], 10*diff_curves_true[0, 1:]*np.log10(np.e) + \
         10*np.log10(diffusion_offsets[0]) + 10*np.log10(counters[0, 1]/counters[0, 0]), \
         linestyle='--', color='tab:blue')
plt.plot(1E9*bins[1:], 10*diff_curves_true[1, 1:]*np.log10(np.e) + \
         10*np.log10(diffusion_offsets[1]) + 10*np.log10(counters[1, 1]/counters[1, 0]), \
     linestyle='--', color='tab:orange')
plt.plot(1E9*bins[1:], 10*diff_curves_true[2, 1:]*np.log10(np.e) + \
         + 10*np.log10(diffusion_offsets[2]) + 10*np.log10(counters[1, 1]/counters[1, 0]), \
         linestyle='--', color='tab:green')
plt.ylim(bottom=0)
plt.legend((r'Fit: $\rho_*$ = {}kg/m3'.format(snow_densities[0]), \
            r'Fit: $\rho_*$ = {}kg/m3'.format(snow_densities[1]), \
            r'Fit: $\rho_*$ = {}kg/m3'.format(snow_densities[2]), \
            r'True: $\rho_*$ = {}kg/m3'.format(snow_densities[0]), \
            r'True: $\rho_*$ = {}kg/m3'.format(snow_densities[1]), \
            r'True: $\rho_*$ = {}kg/m3'.format(snow_densities[2])), loc='upper right')
plt.xlabel('Photon travel-time (ns)')
plt.ylabel('Counts')
plt.title(r'Fitted vs. True Diffusion Curves, $r_*$ = {} um, $\lambda$ = {} nm, Various $\rho_*$'.format(density_snow, wavelength*1E9))
plt.show()

folder = './snowDensityTests/'
filename = 'density_{}_wavelength_{}.npy'.format(density_snow, wavelength)

with open(folder + 'bins_'+filename, 'wb') as f:
    np.save(f, bins)
    
with open(folder + 'hists_'+filename, 'wb') as f:
    np.save(f, counters)  
    
with open(folder + 'ptrue_'+filename, 'wb') as f:
    np.save(f, counters)  
    
with open(folder + 'pfit_'+filename, 'wb') as f:
    np.save(f, counters)  


# %%  Density inversions 532 nm

r_lims = [10E-6, 2E-3] # m
num_r = 200

rho_lims = [50, 750] # kg/m3
num_rho = 200

a1s, a2s, rs, rhos = snow.diffusionParameterLookupTable(wavelength, m_ice, \
                                            r_lims, num_r, rho_lims, num_rho)
    
loss_a1_true = np.zeros((3, num_rho))
rho_est_true = np.zeros(3)
loss_a1_fit = np.zeros_like(loss_a1_true)
rho_est_fit = np.zeros_like(rho_est_true)
    
for ii, density_snow in enumerate(snow_densities):  
    loss_a1_true[ii] = np.sum((a1s-p_true[ii, 1])**2/(a1s**2), axis=0)
    rho_est_true[ii] = rhos[np.argmin(loss_a1_true[ii, :])]
    
    loss_a1_fit[ii] = np.sum((a1s-p_fit[ii, 1])**2/(a1s**2), axis=0)
    rho_est_fit[ii] = rhos[np.argmin(loss_a1_fit[ii, :])]
    
    print('True density = {}, True fit density = {}, Fit density = {}'.format(density_snow, rho_est_true[ii], rho_est_fit[ii]))
        
plt.plot(rhos, np.log(loss_a1_fit[0, :]));
plt.plot(rhos, np.log(loss_a1_fit[1, :]));
plt.plot(rhos, np.log(loss_a1_fit[2, :]));
plt.axvline(snow_densities[0], linestyle='--', color='tab:blue')
plt.axvline(snow_densities[1], linestyle='--', color='tab:orange')
plt.axvline(snow_densities[2], linestyle='--', color='tab:green')
plt.xlabel('Snow density (kg/m3)')
plt.ylabel('log(loss)')
plt.title('Density Estimation Loss Function')
plt.legend((r'{} kg/m3'.format(snow_densities[0]), \
            r'{} kg/m3'.format(snow_densities[1]), \
            r'{} kg/m3'.format(snow_densities[2]) ) )
    
# %% Load data and look at the histograms to choose bin size/window for plots 
# in the next cell
# Radius tests 1064 nm

wavelength = 1064.0E-9
m_ice = m_ice_1064
snow_densities = [80., 250., 500.]
r_snow = 200E-6

for ii, density_snow in enumerate(snow_densities):
    
    print('Simulation completed for snow density = {}'.format(density_snow))
    
    folder = './snowDensityTests/'
    filename = 'grain_radius_{}_density_{}_wavelength_{}.npy'.format(r_snow, density_snow, wavelength)
    
    with open(folder + 'data_' + filename, 'rb') as f:
        batch_data = np.load(f)
        
    data[ii, :] = batch_data[:, 1]
        
    with open(folder + 'counters_' + filename, 'rb') as f:
        counters[ii, :] = np.load(f)
        
    plt.hist(data[ii, :], bins=500, log='True');
    plt.show()
        
    print('Data loaded for snow r = {}, rho = {}'.format(r_snow, density_snow))
    print('Albedo measured to be {}'.format(counters[ii, 1]/counters[ii, 0]))
    
# %%  Plot histograms, diffusion curves and apply fits
# Radius tests 1064 nm

max_travel_dist = 3. # meters
num_bins = 500

bins = np.linspace(0, max_travel_dist/c, num_bins)

hists = np.zeros((3, num_bins))
diff_curves_true = np.zeros_like(hists)
diffusion_offsets = np.zeros(3)
diff_curves_fit = np.zeros_like(hists)
    
for ii, density_snow in enumerate(snow_densities):
    
    medium_params = snow.snowpackScatterProperties(wavelength, m_ice, r_snow, \
                    density_snow, m_soot, r_soot, f_soot, pfunsmoothing=False, mie=False)

    muSs[ii] = medium_params['muS']
    muAs[ii] = medium_params['muA']
    gs[ii] = medium_params['g']
    c_snows[ii] = medium_params['c_snow']
        
    #bins[ii, :] = np.linspace(0, max_travel_dist./c_snows[ii], num_bins)
    hists[ii, :], _ = np.histogram(data[ii, :]/c_snows[ii], bins=num_bins, \
                                range=(0, max_travel_dist/c))
    z0 = 1/(muSs[ii]*(1-gs[ii]))
    D = 1/(3*(muAs[ii] + (1-gs[ii])*muSs[ii]))
    
    p_true[ii, 0] = np.log(z0/np.sqrt(4*np.pi*D*c_snows[ii]))
    p_true[ii, 1] = -muAs[ii]*c_snows[ii]
    p_true[ii, 2] = -z0**2/(4*D*c_snows[ii])
    diff_curves_true[ii, :] = p_true[ii, 0] + p_true[ii, 1]*bins + p_true[ii, 2]/bins + p_true[ii, 3]*np.log(bins)
    diffusion_offsets[ii] = np.sum(hists[ii, 3:])/np.nansum(np.exp(diff_curves_true[ii, 3:]))
        
    p_fit[ii, :], _ = snow.fitLargeFootprintDiffusionCurve(bins[1:], hists[ii, 1:])
    diff_curves_fit[ii, :] = p_fit[ii, 0] + p_fit[ii, 1]*bins + p_fit[ii, 2]/bins + p_fit[ii, 3]*np.log(bins)
    
    print('Snow r = {} um, rho = {} kg/m3, wavelength = {} nm .'.format(r_snow*1E6, density_snow, wavelength*1E9))
    print('True diffusion parameters: {}'.format(p_true[ii, :]))
    print('Fitted diffusion parameters: {}'.format(p_fit[ii, :]))
    
# Time-of-flight histograms + Diffusion curves
plt.plot(1E9*bins, 10*np.log10(hists[0, :]*counters[0, 1]/counters[0, 0])); 
plt.plot(1E9*bins, 10*np.log10(hists[1, :]*counters[1, 1]/counters[1, 0]));
plt.plot(1E9*bins, 10*np.log10(hists[2, :]*counters[2, 1]/counters[2, 0]));
plt.plot(1E9*bins[1:], 10*diff_curves_true[0, 1:]*np.log10(np.e) + \
         10*np.log10(diffusion_offsets[0]) + 10*np.log10(counters[0, 1]/counters[0, 0]), \
         linestyle='--', color='tab:blue')
plt.plot(1E9*bins[1:], 10*diff_curves_true[1, 1:]*np.log10(np.e) + \
         10*np.log10(diffusion_offsets[1]) + 10*np.log10(counters[1, 1]/counters[1, 0]), \
     linestyle='--', color='tab:orange')
plt.plot(1E9*bins[1:], 10*diff_curves_true[2, 1:]*np.log10(np.e) + \
         + 10*np.log10(diffusion_offsets[2]) + 10*np.log10(counters[1, 1]/counters[1, 0]), \
         linestyle='--', color='tab:green')
plt.ylim(bottom=0)
plt.legend((r'Counts: $\rho_*$ = {}kg/m3'.format(snow_densities[0]), \
            r'Counts: $\rho_*$ = {}kg/m3'.format(snow_densities[1]), \
            r'Counts: $\rho_*$ = {}kg/m3'.format(snow_densities[2]), \
            r'Diffusion: $\rho_*$ = {}kg/m3'.format(snow_densities[0]), \
            r'Diffusion: $\rho_*$ = {}kg/m3'.format(snow_densities[1]), \
            r'Diffusion: $\rho_*$ = {}kg/m3'.format(snow_densities[2])), loc='upper right')
plt.xlabel('Photon travel-time (ns)')
plt.ylabel('Counts')
plt.title(r'Time-of-flght histograms, $r_*$ = {} um, $\lambda$ = {} nm, Various $\rho_*$'.format(density_snow, wavelength*1E9))
plt.show()

# Fitted vs True diffusion curves
plt.plot(1E9*bins[1:], 10*diff_curves_fit[0, 1:]*np.log10(np.e)+10*np.log10(counters[0, 1]/counters[0, 0])); 
plt.plot(1E9*bins[1:], 10*diff_curves_fit[1, 1:]*np.log10(np.e) +10*np.log10(counters[1, 1]/counters[1, 0]));
plt.plot(1E9*bins[1:], 10*diff_curves_fit[2, 1:]*np.log10(np.e) + 10*np.log10(counters[2, 1]/counters[2, 0]));
plt.plot(1E9*bins[1:], 10*diff_curves_true[0, 1:]*np.log10(np.e) + \
         10*np.log10(diffusion_offsets[0]) + 10*np.log10(counters[0, 1]/counters[0, 0]), \
         linestyle='--', color='tab:blue')
plt.plot(1E9*bins[1:], 10*diff_curves_true[1, 1:]*np.log10(np.e) + \
         10*np.log10(diffusion_offsets[1]) + 10*np.log10(counters[1, 1]/counters[1, 0]), \
     linestyle='--', color='tab:orange')
plt.plot(1E9*bins[1:], 10*diff_curves_true[2, 1:]*np.log10(np.e) + \
         + 10*np.log10(diffusion_offsets[2]) + 10*np.log10(counters[1, 1]/counters[1, 0]), \
         linestyle='--', color='tab:green')
plt.ylim(bottom=0)
plt.legend((r'Fit: $\rho_*$ = {}kg/m3'.format(snow_densities[0]), \
            r'Fit: $\rho_*$ = {}kg/m3'.format(snow_densities[1]), \
            r'Fit: $\rho_*$ = {}kg/m3'.format(snow_densities[2]), \
            r'True: $\rho_*$ = {}kg/m3'.format(snow_densities[0]), \
            r'True: $\rho_*$ = {}kg/m3'.format(snow_densities[1]), \
            r'True: $\rho_*$ = {}kg/m3'.format(snow_densities[2])), loc='upper right')
plt.xlabel('Photon travel-time (ns)')
plt.ylabel('Counts')
plt.title(r'Fitted vs. True Diffusion Curves, $r_*$ = {} um, $\lambda$ = {} nm, Various $\rho_*$'.format(density_snow, wavelength*1E9))
plt.show()

folder = './snowDensityTests/'
filename = 'density_{}_wavelength_{}.npy'.format(density_snow, wavelength)

with open(folder + 'bins_'+filename, 'wb') as f:
    np.save(f, bins)
    
with open(folder + 'hists_'+filename, 'wb') as f:
    np.save(f, counters)  
    
with open(folder + 'ptrue_'+filename, 'wb') as f:
    np.save(f, counters)  
    
with open(folder + 'pfit_'+filename, 'wb') as f:
    np.save(f, counters)  


# %%  Density inversions 1064 nm

r_lims = [10E-6, 2E-3] # m
num_r = 200

rho_lims = [50, 750] # kg/m3
num_rho = 200

a1s, a2s, rs, rhos = snow.diffusionParameterLookupTable(wavelength, m_ice, \
                                            r_lims, num_r, rho_lims, num_rho)
    
loss_a1_true = np.zeros((3, num_rho))
rho_est_true = np.zeros(3)
loss_a1_fit = np.zeros_like(loss_a1_true)
rho_est_fit = np.zeros_like(rho_est_true)
    
for ii, density_snow in enumerate(snow_densities):  
    loss_a1_true[ii] = np.sum((a1s-p_true[ii, 1])**2/(a1s**2), axis=0)
    rho_est_true[ii] = rhos[np.argmin(loss_a1_true[ii, :])]
    
    loss_a1_fit[ii] = np.sum((a1s-p_fit[ii, 1])**2/(a1s**2), axis=0)
    rho_est_fit[ii] = rhos[np.argmin(loss_a1_fit[ii, :])]
    
    print('True density = {}, True fit density = {}, Fit density = {}'.format(density_snow, rho_est_true[ii], rho_est_fit[ii]))
        
plt.plot(rhos, np.log(loss_a1_fit[0, :]));
plt.plot(rhos, np.log(loss_a1_fit[1, :]));
plt.plot(rhos, np.log(loss_a1_fit[2, :]));
plt.axvline(snow_densities[0], linestyle='--', color='tab:blue')
plt.axvline(snow_densities[1], linestyle='--', color='tab:orange')
plt.axvline(snow_densities[2], linestyle='--', color='tab:green')
plt.xlabel('Snow density (kg/m3)')
plt.ylabel('log(loss)')
plt.title('Density Estimation Loss Function')
plt.legend((r'{} kg/m3'.format(snow_densities[0]), \
            r'{} kg/m3'.format(snow_densities[1]), \
            r'{} kg/m3'.format(snow_densities[2]) ) )


    