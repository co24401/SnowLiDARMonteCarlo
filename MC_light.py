#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import sys
import math
import scipy.interpolate as interpolate

# run_d_MC() is the main wrapper function for the Monte Carlo simulation.
# It accepts all the highlevel simulation parameters and divides the workload to batches.
# Batches are maintained in CPU.

# Input arguments:

# simulation_params = A dictionary that contains parameters that control simulation.
#                     Has the following entries:
#   device_id       = The GPU device ID to run on. 
#   batchSize       = Max number of photons per device (affected by memory considerations)
#   nPhotonsRequested = The size of the array asked to be returned - how many photons should be detected
#   nPhotonsToRun   = The maximum photons to run before giving up
#   max_N           = Maximum number of scattering events before a photon is terminated.
#   max_distance_from_det = Farthest distance from the detector a photon is allowed before it is terminated.

#   muS             = Scattering coefficient (1/m)
#   muA             = Absorption coefficient (1/m)
#   g               =  Anisotropy factor
#   z_bounded       = If True, medium has finite thickness.  Else is semi-infinite.
#   z_range         = [min z (top surface), and max z (bottom surface)] of medium.
#                        Only used if medium is finite

# medium_params     = A dictionary defining the scattering medium properties.
#                     Has the following entries:
#   muS             = Scattering coefficient (1/m)
#   muA             = Absorption coefficient (1/m)
#   g               =  Anisotropy factor
#   z_bounded       = If True, medium has finite thickness.  Else is semi-infinite.
#   z_range         = [min z (top surface), and max z (bottom surface)] of medium.
#                        Only used if medium is finite

# source_params     = A dictionary defining the source.
#                     Has the following entries:
#    r              = Source location (x,y,z) coordinates
#    mu             = Source direction cosines (mu_x, mu_y, mu_z)
#    radius         = Radius of beam footprint (assumed circular)

# detector_params   = A dictionary defining the detector.
#                     Has the following entries:
#    radius         = Radius of detector face.  Assume circular detector.

# Output arguments:
# 
#  data             = A 2d array. Rows are individual detected photons.
#                     columns:
#                       0, 1, 2, 3, 4,   5,    6,    7,    8
#                       n, d, x ,y, z, mu_x, mu_y, mu_z, weight
#
#  photon_counters  = A list with the following entries:
#    0: Number of simulated photons
#    1: Number of detected photons
#    2: Number of photons that didn't get back to the detector because of some of the stopping criteria

def run_d_MC(simulation_params = {'device_id': 0, 'batchSize': 3225600, 
                                  'nPhotonsRequested': 1e7, 
                                  'nPhotonsToRun': 1e10, 
                                  'max_N': 1e5, 
                                  'max_distance_from_det': 100.0}, 
             medium_params = {'muS': 820.0, 'muA': .00820, 'g': 0.89, 
                              'mie': False,
                              'z_bounded': False, 
                              'z_range': np.array([0. , 10.])}, 
             source_params = {'pos': np.array([0., 0., 0.]), 
                              'mu': np.array([0., 0., 1.]), 
                              'radius': 1.},
             detector_params = {'radius': 1.}
                        ):
    
    # Simulation params
    device_id = int(simulation_params['device_id'])
    batchSize = int(simulation_params['batchSize'])
    nPhotonsRequested = int(simulation_params['nPhotonsRequested'])
    nPhotonsToRun = int(simulation_params['nPhotonsToRun'])
    max_N = int(simulation_params['max_N'])
    max_distance_from_det = float(simulation_params['max_distance_from_det'])

    # Medium params
    muS = float(medium_params['muS'])  # Scattering coefficient (1/m)
    muA = float(medium_params['muA'])  # Absorption coefficient (1/m)
    g = float(medium_params['g']) # Anisotropy factor
    z_bounded = bool(medium_params['z_bounded']) # If True, medium has finite thickness
    z_range = medium_params['z_range'].astype(float) # min z and max z
    mie = medium_params['mie']
    if mie:
        inv_cdf_P = medium_params['inv_cdf_P']
    
    # Illumination source params
    source_pos = source_params['pos'].astype(float) # Beam footprint center
    source_mu = source_params['mu'].astype(float) # Direction cosines describing direction of incidence of beam
    source_radius = float(source_params['radius'])  # Beam footprint radius (m)
    
    # Detector parameters
    detector_radius = float(detector_params['radius']) # Detector footprint radius (m)

    data = np.ndarray(shape=(nPhotonsRequested, 9), dtype=float)
    photon_counters = np.zeros(shape=3, dtype=int)
    
    #   photon_counters description:
    #   0: Total simulated photons
    #   1: Detected photons
    #   2: Photons that didn't get back to the detector because of some of the stopping criteria

    # These numbers can be optimized based on the device / architecture / number of photons
    blockSize = 256 # Threads per block.  Should be multiple of 32.
    gridSize = 126 # Blocks per grid.  Should be multiple of # of SMs on GPU
    photons_per_thread = int(np.ceil(float(batchSize)/(blockSize * gridSize)))
    max_photons_per_thread = int(np.ceil(float(nPhotonsToRun)/(blockSize * gridSize)))

    cuda.select_device(device_id)
    device = cuda.get_current_device()
    stream = cuda.stream() 

    while photon_counters[0] < nPhotonsToRun and photon_counters[1] < nPhotonsRequested:
        
        batch_data = np.zeros(shape=(blockSize*gridSize, photons_per_thread, 9), dtype=np.float32)
        batch_counters = np.zeros(shape=(blockSize*gridSize, 3), dtype=np.int)
        batch_data_device = cuda.device_array_like(batch_data, stream=stream)
        batch_counters_device = cuda.device_array_like(batch_counters, stream=stream)

        # Used to initialize the random states in each thread.
        rng_states = create_xoroshiro128p_states(blockSize * gridSize, seed=(np.random.randint(sys.maxsize, dtype=np.int64)-128)+device_id, stream=stream)
        
        if mie:
            # Kernel call
            propPhotonMieGPU[gridSize, blockSize](rng_states, batch_data_device, batch_counters_device,
                                                     photons_per_thread, max_photons_per_thread,
                                                     max_N, max_distance_from_det,
                                                     muS, muA, inv_cdf_P, z_bounded, z_range,
                                                     source_pos, source_mu, source_radius,
                                                     detector_radius)            
        
        else:
            # Kernel call
            propPhotonGPU[gridSize, blockSize](rng_states, batch_data_device, batch_counters_device,
                                                     photons_per_thread, max_photons_per_thread,
                                                     max_N, max_distance_from_det,
                                                     muS, muA, g, z_bounded, z_range,
                                                     source_pos, source_mu, source_radius,
                                                     detector_radius)
        
        # Copy data back
        batch_data_device.copy_to_host(batch_data, stream=stream)
        batch_counters_device.copy_to_host(batch_counters, stream=stream)
        stream.synchronize() # Don't continue until data copied to host
        
        batch_data = batch_data.reshape(batch_data.shape[0]*batch_data.shape[1], batch_data.shape[2])        
        batch_counters_aggr = np.squeeze(np.sum(batch_counters, axis=0))
        
        batch_data = batch_data[batch_data[:, 0]>0, :]

        if batch_data.shape[0] > nPhotonsRequested - photon_counters[1]:
            batch_data = batch_data[:nPhotonsRequested - photon_counters[1], :]
            
        data[photon_counters[1] : (photon_counters[1]+batch_data.shape[0]), :] = batch_data

        photon_counters += batch_counters_aggr

    data = data[:photon_counters[1], :]
    return data, photon_counters

# propPhotonGPU() is the MC GPU kernel function
#
# Input
# =====
#  photons_requested: total photons in the detector (max returned in the array)
#  max_photons_to_run: maximum of the photons that are allowed to simulate
#
#  Return columns:
#    0, 1, 2, 3, 4,   5,    6,    7,    8
#    n, d, x ,y, z, mu_x, mu_y, mu_z, weight
#
#   photon_counters:
#    0: Total simulated photons
#    1: Detected photons
#    2: Photons that didn't get back to the detector because of some of the stopping criteria

@cuda.jit
def propPhotonGPU(rng_states, data_out, photon_counters,
                  photons_requested, max_photons_to_run, 
                  max_N, max_distance_from_det,
                  muS, muA, g, z_bounded, z_range,
                  source_pos, source_mu, source_radius,
                  detector_radius):
    # Setup
    thread_id = cuda.grid(1)
    z_min, z_max = z_range[0], z_range[1]
    detR2 = detector_radius**2
    albedo = muS / (muS + muA)
    
    w_threshold = 0.0001 # Termination threshold for roulette
    roulette_factor = 10

    photon_cnt_tot = 0          # Total simulated photons
    photons_cnt_stopped = 0     # Photons that didn't get back to the detector because of some of the stopping criteria
    photons_cnt_detected = 0    # Detected photons

    # Main loop over the number of required photons.
    while photon_cnt_tot < max_photons_to_run and photons_cnt_detected < photons_requested:
        photon_cnt_tot += 1
        
        w = 1.

        data_out[thread_id, photons_cnt_detected, :] = -1.0

        # If area source, uniformly sample random point within circular beam footprint
        if source_radius > 0:
            rand1 = xoroshiro128p_uniform_float32(rng_states, thread_id)        
            rand2 = xoroshiro128p_uniform_float32(rng_states, thread_id)
            x = source_radius*math.sqrt(rand1)*math.cos(2*math.pi*rand2) + source_pos[0]
            y = source_radius*math.sqrt(rand1)*math.sin(2*math.pi*rand2) + source_pos[1]
        else: # Fixed x,y,z (pencil)
            x, y =  source_pos[0], source_pos[1]
        z = source_pos[2]
        nux, nuy, nuz = source_mu[0], source_mu[1], source_mu[2]

        d = 0.
        n = 0

        # Start Monte Carlo inifinite loop for traced photon
        while True:
            # Should we stop?
            if n >= max_N:
                photons_cnt_stopped += 1
                break
            if math.sqrt(x*x + y*y + z*z) > max_distance_from_det:  # Assumes detector at origin
                photons_cnt_stopped += 1
                break
            if z_bounded:# Check if we are out of tissue (when starting from tissue z boundary)
                if z > z_max or z < z_min:
                    photons_cnt_stopped += 1
                    break

            # Get random numbers
            rand1 = xoroshiro128p_uniform_float32(rng_states, thread_id)
            rand2 = xoroshiro128p_uniform_float32(rng_states, thread_id)
            rand3 = xoroshiro128p_uniform_float32(rng_states, thread_id)

            # Calculate random propogation distance
            cd = - math.log(rand1) / muS

            # Update temporary new location
            t_rx = x + cd * nux
            t_ry = y + cd * nuy
            t_rz = z + cd * nuz

            # Check if we hit the detector
            if t_rz <= 0: # Did we pass the detector?
                cd = - z / nuz

                t_rx = x + cd * nux
                t_ry = y + cd * nuy
                t_rz = z + cd * nuz

                if t_rx**2 + t_ry**2 < detR2: # If we hit the aperture
                    # Photon was detected
                    d+=cd
                    n+=1
                    x,y,z = t_rx, t_ry, t_rz
                    
                    w = albedo*w
                    if w < w_threshold:
                        rand_roulette = xoroshiro128p_uniform_float32(rng_states, thread_id)
                        if rand_roulette <= 1/roulette_factor:
                            w = roulette_factor*w
                        else:
                            photons_cnt_stopped += 1
                            break

                    # Record data and break
                    data_out[thread_id, photons_cnt_detected, 0] = n
                    data_out[thread_id, photons_cnt_detected, 1] = d
                    data_out[thread_id, photons_cnt_detected, 2] = x
                    data_out[thread_id, photons_cnt_detected, 3] = y
                    data_out[thread_id, photons_cnt_detected, 4] = z
                    data_out[thread_id, photons_cnt_detected, 5] = nux
                    data_out[thread_id, photons_cnt_detected, 6] = nuy
                    data_out[thread_id, photons_cnt_detected, 7] = nuz
                    data_out[thread_id, photons_cnt_detected, 8] = w
                    photons_cnt_detected += 1
                    break
                else:  # If we passed the detector and didn't hit it we should stop
                    photons_cnt_stopped += 1
                    break

            # Update photon
            x, y, z = t_rx, t_ry, t_rz
            d += cd #prop distance
            n += 1 #increase scatter counter
            
            w = albedo*w
            if w < w_threshold:
                rand_roulette = xoroshiro128p_uniform_float32(rng_states, thread_id)
                if rand_roulette <= 1/roulette_factor:
                    w = roulette_factor*w
                else:
                    photons_cnt_stopped += 1
                    break

            # Scatter to new angle
            psi = 2 * math.pi * rand2
            mu = 1/(2*g) * (1+g**2 - ( (1-g*g)/(1-g+2*g*rand3))**2)

            # Update angels
            sin_psi = math.sin(psi)
            cos_psi = math.cos(psi)
            sqrt_mu = math.sqrt(1-mu**2)
            sqrt_w  = math.sqrt(1-nuz**2)
            if sqrt_w != 0:
                prev_nux, prev_nuy, prev_nuz = nux, nuy, nuz
                nux = prev_nux*mu + (prev_nux*prev_nuz*cos_psi - prev_nuy*sin_psi)*sqrt_mu/sqrt_w
                nuy = prev_nuy*mu + (prev_nuy*prev_nuz*cos_psi + prev_nux*sin_psi)*sqrt_mu/sqrt_w
                nuz = prev_nuz*mu - cos_psi*sqrt_mu*sqrt_w
            elif nuz==1.0:
                nux = sqrt_mu*cos_psi
                nuy = sqrt_mu*sin_psi
                nuz = mu
            else: # nu[2]==-1.0
                nux = sqrt_mu*cos_psi
                nuy = -sqrt_mu*sin_psi
                nuz = -mu

    # Update photon counters before completion
    photon_counters[thread_id, 0] = photon_cnt_tot
    photon_counters[thread_id, 1] = photons_cnt_detected
    photon_counters[thread_id, 2] = photons_cnt_stopped
    
# propPhotonMieGPU() is the MC GPU kernel function
#
# Input
# =====
#  photons_requested: total photons in the detector (max returned in the array)
#  max_photons_to_run: maximum of the photons that are allowed to simulate
#
#  Return columns:
#    0, 1, 2, 3, 4,   5,    6,    7,    8
#    n, d, x ,y, z, mu_x, mu_y, mu_z, weight
#
#   photon_counters:
#    0: Total simulated photons
#    1: Detected photons
#    2: Photons that didn't get back to the detector because of some of the stopping criteria

@cuda.jit
def propPhotonMieGPU(rng_states, data_out, photon_counters,
                  photons_requested, max_photons_to_run, 
                  max_N, max_distance_from_det,
                  muS, muA, inv_cdf_P, z_bounded, z_range,
                  source_pos, source_mu, source_radius,
                  detector_radius):
    # Setup
    thread_id = cuda.grid(1)
    z_min, z_max = z_range[0], z_range[1]
    detR2 = detector_radius**2
    albedo = muS / (muS + muA)
    
    w_threshold = 0.0001 # Termination threshold for roulette
    roulette_factor = 10

    photon_cnt_tot = 0          # Total simulated photons
    photons_cnt_stopped = 0     # Photons that didn't get back to the detector because of some of the stopping criteria
    photons_cnt_detected = 0    # Detected photons

    # Main loop over the number of required photons.
    while photon_cnt_tot < max_photons_to_run and photons_cnt_detected < photons_requested:
        photon_cnt_tot += 1
        
        w = 1.

        data_out[thread_id, photons_cnt_detected, :] = -1.0

        # If area source, uniformly sample random point within circular beam footprint
        if source_radius > 0:
            rand1 = xoroshiro128p_uniform_float32(rng_states, thread_id)        
            rand2 = xoroshiro128p_uniform_float32(rng_states, thread_id)
            x = source_radius*math.sqrt(rand1)*math.cos(2*math.pi*rand2) + source_pos[0]
            y = source_radius*math.sqrt(rand1)*math.sin(2*math.pi*rand2) + source_pos[1]
        else: # Fixed x,y,z (pencil)
            x, y =  source_pos[0], source_pos[1]
        z = source_pos[2]
        nux, nuy, nuz = source_mu[0], source_mu[1], source_mu[2]

        d = 0.
        n = 0

        # Start Monte Carlo inifinite loop for traced photon
        while True:
            # Should we stop?
            if n >= max_N:
                photons_cnt_stopped += 1
                break
            if math.sqrt(x*x + y*y + z*z) > max_distance_from_det:  # Assumes detector at origin
                photons_cnt_stopped += 1
                break
            if z_bounded:# Check if we are out of tissue (when starting from tissue z boundary)
                if z > z_max or z < z_min:
                    photons_cnt_stopped += 1
                    break

            # Get random numbers
            rand1 = xoroshiro128p_uniform_float32(rng_states, thread_id)
            rand2 = xoroshiro128p_uniform_float32(rng_states, thread_id)
            rand3 = xoroshiro128p_uniform_float32(rng_states, thread_id)

            # Calculate random propogation distance
            cd = - math.log(rand1) / muS

            # Update temporary new location
            t_rx = x + cd * nux
            t_ry = y + cd * nuy
            t_rz = z + cd * nuz

            # Check if we hit the detector
            if t_rz <= 0: # Did we pass the detector?
                cd = - z / nuz

                t_rx = x + cd * nux
                t_ry = y + cd * nuy
                t_rz = z + cd * nuz

                if t_rx**2 + t_ry**2 < detR2: # If we hit the aperture
                    # Photon was detected
                    d+=cd
                    n+=1
                    x,y,z = t_rx, t_ry, t_rz
                    
                    w = albedo*w
                    if w < w_threshold:
                        rand_roulette = xoroshiro128p_uniform_float32(rng_states, thread_id)
                        if rand_roulette <= 1/roulette_factor:
                            w = roulette_factor*w
                        else:
                            photons_cnt_stopped += 1
                            break

                    # Record data and break
                    data_out[thread_id, photons_cnt_detected, 0] = n
                    data_out[thread_id, photons_cnt_detected, 1] = d
                    data_out[thread_id, photons_cnt_detected, 2] = x
                    data_out[thread_id, photons_cnt_detected, 3] = y
                    data_out[thread_id, photons_cnt_detected, 4] = z
                    data_out[thread_id, photons_cnt_detected, 5] = nux
                    data_out[thread_id, photons_cnt_detected, 6] = nuy
                    data_out[thread_id, photons_cnt_detected, 7] = nuz
                    data_out[thread_id, photons_cnt_detected, 8] = w
                    photons_cnt_detected += 1
                    break
                else:  # If we passed the detector and didn't hit it we should stop
                    photons_cnt_stopped += 1
                    break

            # Update photon
            x, y, z = t_rx, t_ry, t_rz
            d += cd #prop distance
            n += 1 #increase scatter counter
            
            w = albedo*w
            if w < w_threshold:
                rand_roulette = xoroshiro128p_uniform_float32(rng_states, thread_id)
                if rand_roulette <= 1/roulette_factor:
                    w = roulette_factor*w
                else:
                    photons_cnt_stopped += 1
                    break

            # Scatter to new angle
            psi = 2 * math.pi * rand2
            mu = inv_cdf_P(rand3)

            # Update angels
            sin_psi = math.sin(psi)
            cos_psi = math.cos(psi)
            sqrt_mu = math.sqrt(1-mu**2)
            sqrt_w  = math.sqrt(1-nuz**2)
            if sqrt_w != 0:
                prev_nux, prev_nuy, prev_nuz = nux, nuy, nuz
                nux = prev_nux*mu + (prev_nux*prev_nuz*cos_psi - prev_nuy*sin_psi)*sqrt_mu/sqrt_w
                nuy = prev_nuy*mu + (prev_nuy*prev_nuz*cos_psi + prev_nux*sin_psi)*sqrt_mu/sqrt_w
                nuz = prev_nuz*mu - cos_psi*sqrt_mu*sqrt_w
            elif nuz==1.0:
                nux = sqrt_mu*cos_psi
                nuy = sqrt_mu*sin_psi
                nuz = mu
            else: # nu[2]==-1.0
                nux = sqrt_mu*cos_psi
                nuy = -sqrt_mu*sin_psi
                nuz = -mu

    # Update photon counters before completion
    photon_counters[thread_id, 0] = photon_cnt_tot
    photon_counters[thread_id, 1] = photons_cnt_detected
    photon_counters[thread_id, 2] = photons_cnt_stopped
