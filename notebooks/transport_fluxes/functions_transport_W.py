import numpy as np
import netCDF4 as nc
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_variables_planes(dirname, filename):
    
    with nc.Dataset(os.path.join(dirname, filename), 'r') as nbl:
        x, y =  slice(1,-1,None), slice(1,-1,None)
        gdepw = nbl.variables['gdepw_1d'][0, :]
        tmask = nbl.variables['tmask'][0, :, y, x]
        mbathy = nbl.variables['mbathy'][0, y, x]
        e1t = nbl.variables['e1t'][0, y, x]
        e2t = nbl.variables['e2t'][0, y, x]  
        
        return gdepw, tmask, mbathy, e1t, e2t

# ------------------------------------------------------------------------------------------------

def build_surface_mask(tmask_nocanyon, mbathy_nocanyon):
    
    surface_mask = np.zeros_like(tmask_nocanyon)
    for i in range(tmask_nocanyon.shape[-1]):
        for j in range(tmask_nocanyon.shape[-2]):
            k = mbathy_nocanyon[j, i]
            surface_mask[k, j, i] = 1
            
    return surface_mask

# ------------------------------------------------------------------------------------------------