# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:17:11 2020

@author: dolor
"""

import numpy as np

def fitLargeFootprintDiffusionCurve(t, m, a3constraint = False):
    
    y = np.log(m) # Fit to log of photon counts
    w = m # Input m are Poission counts.  Weights 1/sigma_y^2 = m
    
    # Defining functions explicitly for readaility and so that can fit to 
    # different sets of functions in future
    f0 = np.ones_like(t)
    f1 = t
    f2 = 1./t
    f3 = np.log(t)
    
    if a3constraint:
        y = y + 1.5*f3
        b = np.array([np.nansum(y*f0*w), \
                      np.nansum(y*f1*w), \
                      np.nansum(y*f2*w)] )
            
        aa00 = np.nansum(w*f0**2)
        aa01 = np.nansum(w*f1*f0)
        aa02 = np.nansum(w*f2*f0)
        
        aa11 = np.nansum(w*f1**2)
        aa12 = np.nansum(w*f1*f2)
        
        aa22 = np.nansum(w*f2**2)
        
        A = np.array([[aa00, aa01, aa02], [aa01, aa11, aa12], \
                      [aa02, aa12, aa22]])
            
        V = np.linalg.inv(A)
        p = V@b
        
    else:
        b = np.array([np.nansum(y*f0*w), \
                      np.nansum(y*f1*w), \
                      np.nansum(y*f2*w), \
                      np.nansum(y*f3*w)] )
            
        aa00 = np.nansum(w*f0**2)
        aa01 = np.nansum(w*f1*f0)
        aa02 = np.nansum(w*f2*f0)
        aa03 = np.nansum(w*f3*f0)
        
        aa11 = np.nansum(w*f1**2)
        aa12 = np.nansum(w*f1*f2)
        aa13 = np.nansum(w*f1*f3)
        
        aa22 = np.nansum(w*f2**2)
        aa23 = np.nansum(w*f2*f3)
        
        aa33 = np.nansum(w*f3**2)
        
        A = np.array([[aa00, aa01, aa02, aa03], [aa01, aa11, aa12, aa13], \
                      [aa02, aa12, aa22, aa23], [aa03, aa13, aa23, aa33]])
            
        V = np.linalg.inv(A)
        p = V@b
    
    return p, V