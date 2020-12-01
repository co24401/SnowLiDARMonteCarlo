# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:44:56 2020

@author: dolor
"""

import numpy as np
import math

# propPhotonCPU() is CPU version of propPhotonGPU() that I'm using for debugging purposes
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

def propPhotonCPU(rng, data_out, photon_counters,
                  photons_requested, max_photons_to_run, 
                  max_N, max_distance_from_det,
                  muS, muA, inv_cdf_P, z_bounded, z_range,
                  source_pos, source_mu, source_radius,
                  detector_radius):
    
    # Setup
    z_min, z_max = z_range[0], z_range[1]
    detR2 = detector_radius**2
    albedo = muS / (muS + muA)
    
    w_threshold = 0.001 # Termination threshold for roulette
    roulette_factor = 10

    photon_cnt_tot = 0          # Total simulated photons
    photons_cnt_stopped = 0     # Photons that didn't get back to the detector because of some of the stopping criteria
    photons_cnt_detected = 0    # Detected photons

    # Main loop over the number of required photons.
    while photon_cnt_tot < max_photons_to_run and photons_cnt_detected < photons_requested:
        photon_cnt_tot += 1
        
        w = 1.

        data_out[photons_cnt_detected, :] = -1.0

        # If area source, uniformly sample random point within circular beam footprint
        if source_radius > 0:
            rand1 = rng.random()      
            rand2 = rng.random()
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
            rand1 = rng.random()
            rand2 = rng.random()
            rand3 = rng.random()

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
                        rand_roulette = rng.random()
                        if rand_roulette <= 1/roulette_factor:
                            w = roulette_factor*w
                        else:
                            photons_cnt_stopped += 1
                            break

                    # Record data and break
                    data_out[photons_cnt_detected, 0] = n
                    data_out[photons_cnt_detected, 1] = d
                    data_out[photons_cnt_detected, 2] = x
                    data_out[photons_cnt_detected, 3] = y
                    data_out[photons_cnt_detected, 4] = z
                    data_out[photons_cnt_detected, 5] = nux
                    data_out[photons_cnt_detected, 6] = nuy
                    data_out[photons_cnt_detected, 7] = nuz
                    data_out[photons_cnt_detected, 8] = w
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
                rand_roulette = rng.random()
                if rand_roulette <= 1/roulette_factor:
                    w = roulette_factor*w
                else:
                    photons_cnt_stopped += 1
                    break

            # Scatter to new angle
            psi = 2 * math.pi * rand2
            mu = inv_cdf_P(rand3)
            #mu = 1/(2*g) * (1+g**2 - ( (1-g*g)/(1-g+2*g*rand3))**2)

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
    photon_counters[0] = photon_cnt_tot
    photon_counters[1] = photons_cnt_detected
    photon_counters[2] = photons_cnt_stopped
    