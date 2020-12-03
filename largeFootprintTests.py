# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:03:47 2020

@author: dolor
"""

import MC_snow as snow
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import sys
#from numba import jit

# %% Define medium input parameters

m_ice_532 = 1.3116 + 1.4898e-9j
m_ice_1064 = 1.3004 + 1.9e-6j

wavelength = 532.0E-9

r_ice = 200E-6
m_ice = m_ice_532
density_snow = 250 # kg/m3 For moderately settled snow, but still dry

m_soot = 1.8 + 0.5j # not wavelength dependent in WWII 1981
r_soot = 0.1E-6
f_soot = 0.#1E-6 # density_soot/density_snow (1E-6 = 1 ppmw soot)

pfun_bins = 1000


# %% Generate medium

medium_params_mie = snow.snowpackScatterProperties(wavelength, m_ice, r_ice, \
                    density_snow, m_soot, r_soot, f_soot, pfunsmoothing=False, mie=True)
    
medium_params_HG = snow.snowpackScatterProperties(wavelength, m_ice, r_ice, \
                    density_snow, m_soot, r_soot, f_soot, pfunsmoothing=False, mie=False)
    
print('Snowpack properties generated')

    
# %% Batch CPU test
# Explicitly define required simulation, source, and detector parameters for 
# single batch run

# Simulation params
batchSize = 10000
max_photons_per_thread = 100000
max_N = 1e5
max_distance_from_det = 100.0

# Source params
source_pos = np.array([0., 0., 0.])
source_mu = np.array([0., 0., 1.])
source_radius = 1.

# Detector params
detector_radius = 1.

# Medium params
muS_HG = medium_params_HG['muS']
muA_HG = medium_params_HG['muA']
g_HG = medium_params_HG['g']
z_bounded_HG = medium_params_HG['z_bounded']
z_range_HG = medium_params_HG['z_range']

muS_mie = medium_params_mie['muS']
muA_mie = medium_params_mie['muA']
inv_cdf_P = medium_params_mie['inv_cdf_P']
z_bounded_mie = medium_params_mie['z_bounded']
z_range_mie = medium_params_mie['z_range']


rng = np.random.default_rng()

batch_data_HG = np.zeros(shape=(batchSize, 9), dtype=np.float32)
batch_counters_HG = np.zeros(shape=(3), dtype=np.int)

print('Beginning CPU MC simulations')

tic = time.time()

snow.propPhotonCPU(rng, batch_data_HG, batch_counters_HG,
                                         batchSize, max_photons_per_thread,
                                         max_N, max_distance_from_det,
                                         muS_HG, muA_HG, g_HG, z_bounded_HG, z_range_HG,
                                         source_pos, source_mu, source_radius,
                                         detector_radius)

print(time.time() - tic)

batch_data_mie = np.zeros(shape=(batchSize, 9), dtype=np.float32)
batch_counters_mie = np.zeros(shape=(3), dtype=np.int)

tic = time.time()

snow.propPhotonMieCPU(rng, batch_data_mie, batch_counters_mie,
                                         batchSize, max_photons_per_thread,
                                         max_N, max_distance_from_det,
                                         muS_mie, muA_mie, inv_cdf_P, z_bounded_mie, z_range_mie,
                                         source_pos, source_mu, source_radius,
                                         detector_radius)

print(time.time() - tic)

plt.hist(batch_data_HG[:, 1], bins=int(batchSize/20), log='True', weights=batch_data_HG[:, 8]); 
plt.title('Distance Traveled -- Henyey-Greenstein')
plt.show()

plt.hist(batch_data_mie[:, 1], bins=int(batchSize/20), log='True', weights=batch_data_mie[:, 8]); 
plt.title('Distance Traveled -- Mie')
plt.show()

# %% Batch GPU test

blockSize = 256 # Threads per block.  Should be multiple of 32.
gridSize = 126 # Blocks per grid.  Should be multiple of # of SMs on GPU
photons_per_thread = 10
max_photons_per_thread = 100
batchSize = blockSize*gridSize*photons_per_thread

device_id = 0

# P = medium_params_mie['P']
# bin_edges = np.linspace(-1, 1, 1001)
# cdf_P = np.zeros_like(bin_edges)
# cdf_P[1:] = np.cumsum(P)
# inv_cdf_P = lambda xi: np.interp(xi, cdf_P, bin_edges)

# @jit
# def inv_cdf_P(xi):
#     return np.interp(xi, cdf_P, bin_edges)

cuda.select_device(device_id)
device = cuda.get_current_device()
stream = cuda.stream()

# Mie simulation

batch_data_mie = np.zeros(shape=(blockSize*gridSize, photons_per_thread, 9), dtype=np.float32)
batch_counters_mie = np.zeros(shape=(blockSize*gridSize, 3), dtype=np.int)

batch_data_device_mie = cuda.device_array_like(batch_data_mie, stream=stream)
batch_counters_device_mie = cuda.device_array_like(batch_counters_mie, stream=stream)

# Used to initialize the random states in each thread.
rng_states = create_xoroshiro128p_states(blockSize * gridSize, seed=(np.random.randint(sys.maxsize, dtype=np.int64)-128)+device_id, stream=stream)

tic = time.time()

snow.propPhotonMieGPU[gridSize, blockSize](rng_states, batch_data_device_mie, batch_counters_device_mie,
                                         photons_per_thread, max_photons_per_thread,
                                         max_N, max_distance_from_det,
                                         muS_mie, muA_mie, inv_cdf_P, z_bounded_mie, z_range_mie,
                                         source_pos, source_mu, source_radius,
                                         detector_radius)

print(time.time() - tic)

# Copy data back
batch_data_device_mie.copy_to_host(batch_data_mie, stream=stream)
batch_counters_device_mie.copy_to_host(batch_counters_mie, stream=stream)
stream.synchronize() # Don't continue until data copied to host

batch_data_mie = batch_data_mie.reshape(batch_data_mie.shape[0]*batch_data_mie.shape[1], batch_data_mie.shape[2])        
batch_counters_aggr_mie = np.squeeze(np.sum(batch_counters_mie, axis=0))

batch_data_mie = batch_data_mie[batch_data_mie[:, 0]>0, :]    

# HG simulation

batch_data_HG = np.zeros(shape=(blockSize*gridSize, photons_per_thread, 9), dtype=np.float32)
batch_counters_HG = np.zeros(shape=(blockSize*gridSize, 3), dtype=np.int)

batch_data_device_HG = cuda.device_array_like(batch_data_HG, stream=stream)
batch_counters_device_HG = cuda.device_array_like(batch_counters_HG, stream=stream)

print('Beginning CPU MC simulations')

tic = time.time()

snow.propPhotonGPU[gridSize, blockSize](rng_states, batch_data_device_HG, batch_counters_device_HG,
                                         photons_per_thread, max_photons_per_thread,
                                         max_N, max_distance_from_det,
                                         muS_HG, muA_HG, g_HG, z_bounded_HG, z_range_HG,
                                         source_pos, source_mu, source_radius,
                                         detector_radius)

print(time.time() - tic)

# Copy data back
batch_data_device_HG.copy_to_host(batch_data_HG, stream=stream)
batch_counters_device_HG.copy_to_host(batch_counters_HG, stream=stream)
stream.synchronize() # Don't continue until data copied to host

batch_data_HG = batch_data_HG.reshape(batch_data_HG.shape[0]*batch_data_HG.shape[1], batch_data_HG.shape[2])        
batch_counters_aggr_HG = np.squeeze(np.sum(batch_counters_HG, axis=0))

batch_data_HG = batch_data_HG[batch_data_HG[:, 0]>0, :]    

plt.hist(batch_data_mie[:, 1], bins=1000, log='True', weights=batch_data_mie[:, 8]); 
plt.title('Distance Traveled -- Mie')
plt.show()

plt.hist(batch_data_HG[:, 1], bins=1000, log='True', weights=batch_data_HG[:, 8]); 
plt.title('Distance Traveled -- Henyey-Greenstein')
plt.show()

# %%  Batch CPU Quantized vs weight-based simulations
# Explicitly define required simulation, source, and detector parameters for 
# single batch run

# Simulation params
batchSize = 100000
max_photons_per_thread = 1000000
max_N = 1e5
max_distance_from_det = 100.0

# Source params
source_pos = np.array([0., 0., 0.])
source_mu = np.array([0., 0., 1.])
source_radius = 1.

# Detector params
detector_radius = 1.

# Medium params
muS = medium_params_HG['muS']
muA = medium_params_HG['muA']
g = medium_params_HG['g']
z_bounded = medium_params_HG['z_bounded']
z_range = medium_params_HG['z_range']

rng = np.random.default_rng()

# batch_data_wts = np.zeros(shape=(batchSize, 9), dtype=np.float32)
# batch_counters_wts = np.zeros(shape=(3), dtype=np.int)

# print('Beginning CPU MC simulations')

# tic = time.time()

# snow.propPhotonCPU(rng, batch_data_wts, batch_counters_wts,
#                                          batchSize, max_photons_per_thread,
#                                          max_N, max_distance_from_det,
#                                          muS, muA, g, z_bounded, z_range,
#                                          source_pos, source_mu, source_radius,
#                                          detector_radius)

# print(time.time() - tic)

batch_data_qnt = np.zeros(shape=(batchSize, 9), dtype=np.float32)
batch_counters_qnt = np.zeros(shape=(3), dtype=np.int)

tic = time.time()

snow.propQuantizedPhotonsCPU(rng, batch_data_qnt, batch_counters_qnt,
                                         batchSize, max_photons_per_thread,
                                         max_N, max_distance_from_det,
                                         muS, muA, g, z_bounded, z_range,
                                         source_pos, source_mu, source_radius,
                                         detector_radius)

print(time.time() - tic)

# plt.hist(batch_data_wts[:, 1], bins=200, log='True', weights=batch_data_wts[:, 8]); 
# plt.title('Distance Traveled -- Weighted Photons')
# plt.show()

plt.hist(batch_data_qnt[:, 1], bins=200, log='True'); 
plt.title('Distance Traveled -- Quantized Photons')
plt.show()

# %% Batch GPU test.  Quantized propagation

max_N = 1e6
max_distance_from_det = 1000.0

# Source params
source_pos = np.array([0., 0., 0.])
source_mu = np.array([0., 0., 1.])
source_radius = 0.

# Detector params
detector_radius = 1000.

# Medium params
muS = medium_params_HG['muS']
muA = medium_params_HG['muA']
g = medium_params_HG['g']
z_bounded = medium_params_HG['z_bounded']
z_range = medium_params_HG['z_range']

blockSize = 256 # Threads per block.  Should be multiple of 32.
gridSize = 126 # Blocks per grid.  Should be multiple of # of SMs on GPU
photons_per_thread = 30
max_photons_per_thread = 100
batchSize = blockSize*gridSize*photons_per_thread

device_id = 0

cuda.select_device(device_id)
device = cuda.get_current_device()
stream = cuda.stream()

# HG simulation

batch_data = np.zeros(shape=(blockSize*gridSize, photons_per_thread, 9), dtype=np.float32)
batch_counters = np.zeros(shape=(blockSize*gridSize, 3), dtype=np.int)

batch_data_device = cuda.device_array_like(batch_data, stream=stream)
batch_counters_device = cuda.device_array_like(batch_counters, stream=stream)

print('Beginning GPU MC simulations')

tic = time.time()

snow.propQuantizedPhotonsGPU[gridSize, blockSize](rng_states, batch_data_device, batch_counters_device,
                                         photons_per_thread, max_photons_per_thread,
                                         max_N, max_distance_from_det,
                                         muS, muA, g, z_bounded, z_range,
                                         source_pos, source_mu, source_radius,
                                         detector_radius)

print(time.time() - tic)

# Copy data back
batch_data_device.copy_to_host(batch_data, stream=stream)
batch_counters_device.copy_to_host(batch_counters, stream=stream)
stream.synchronize() # Don't continue until data copied to host

batch_data = batch_data.reshape(batch_data.shape[0]*batch_data.shape[1], batch_data.shape[2])        
batch_counters_aggr = np.squeeze(np.sum(batch_counters, axis=0))

batch_data = batch_data[batch_data[:, 0]>0, :]

plt.hist(batch_data[:, 1], bins=1000, log='True'); 
plt.title('Distance Traveled -- Quantized Photons')
plt.show()






