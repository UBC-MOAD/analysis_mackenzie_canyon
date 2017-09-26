import numpy as np
import netCDF4 as nc
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from salishsea_tools.nc_tools import scDataset
import pickle

import sys
sys.path.append('/ocean/imachuca/Canyons/analysis_mackenzie_canyon/notebooks/general_circulation/')
import general_functions


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

def save_surface_mask(surface_mask):
    
    file_name = 'surface_mask'
    file_object = open(file_name,'wb')
    pickle.dump(surface_mask,file_object)
    file_object.close()
    
    return file_name

def get_surface_mask(tmaskn, mbathyn):
    
    surface_mask = build_surface_mask(tmaskn, mbathyn)
    file_name = save_surface_mask(surface_mask)
    
    file_object = open(file_name,'rb')
    surface_mask_saved = pickle.load(file_object)
    
    return surface_mask_saved

# ------------------------------------------------------------------------------------------------

def get_variables_fluxes(dirname, filepattern, surface_mask):
    
    files = general_functions.get_files(dirname, filepattern, 'grid_W')
    x, y =  slice(1,-1,None), slice(1,-1,None)
    with scDataset(files) as ds:
        vovecrtz0 = ds.variables['vovecrtz'][:, :, y, x]
    surface_mask_f = np.tile(surface_mask, (vovecrtz0.shape[0], 1, 1, 1))
    vovecrtz = np.ma.array(vovecrtz0, mask = 1 - surface_mask_f)

    return vovecrtz

# ------------------------------------------------------------------------------------------------

def get_index_axis(mbathy, mbathyn):

    mbathy_diff = mbathy - mbathyn
    axis_thalweg = np.zeros(mbathy_diff.shape[-2])

    for y in range(len(axis_thalweg)):
        mbathy_row = mbathy_diff[y, :]
        max_mbathy_row = mbathy_row.max()
        x_inds_max_mbathy_row = np.where(mbathy_row == max_mbathy_row)[0]
        x_ind_thalweg = int(np.median(x_inds_max_mbathy_row))
        axis_thalweg[y] = x_ind_thalweg
        
    return axis_thalweg


def get_index_break(gdepw, mbathyn):
    
    # [ 0  5  6  7  8  9 10 20 30 40 50 60 63 65 67 69 72 74 76 78 79]
    unique_z_inds = np.unique(mbathyn[:, 1])

    # [ 4 27 38 38 38 33  3  3  3  3  3  3  3  3  3  3  3  3  3  3 52]
    occurrence_z_inds = np.zeros_like(unique_z_inds)
    for i in range(len(unique_z_inds)):
        occurrence_z_inds[i] = (mbathyn[:, 1] == unique_z_inds[i]).sum()

    # [23 11  0  0  5 30  0  0  0  0  0  0  0  0  0  0  0  0  0 49]
    jumps_in_occurrence = abs(np.diff(occurrence_z_inds))

    # [array([ 1,  2,  6, 20])]
    con1 = [x+1 for x in np.where(jumps_in_occurrence > np.mean(jumps_in_occurrence))] 

    # [ 5  6 10 79]
    unique_z_inds_con1 = unique_z_inds[con1]

    # [   43.75    52.5     87.5   1300.  ]
    dep_z_inds_con1 = gdepw[unique_z_inds_con1]

    # 2
    con2 = np.argmin(np.abs(dep_z_inds_con1 - 80))

    # 10
    unique_z_inds_con1_con2 = unique_z_inds_con1[con2]

    # 178
    ind_break = np.where(mbathyn[:, 1] == unique_z_inds_con1_con2)[0][0]
    
    return ind_break


def get_index_head(gdepw, tmask, axis_thalweg):
    
    ind_shelf = np.argmin(np.abs(gdepw - 80))

    state_thalweg = np.zeros_like(axis_thalweg)
    for i in range(len(axis_thalweg)):
        state_thalweg[i] = tmask[ind_shelf, i, axis_thalweg[i]]

    ind_head = np.where(state_thalweg == 0)[0][-1]
    
    return ind_head, ind_shelf


def get_index_half(tmask, e1t, e2t, surface_mask):
    
    area_k0 = e1t * e2t
    area_k = area_k0 * surface_mask

    # 1. area of individual wet cells
    area_for_mid = area_k * tmask

    area_k0.shape, area_k.shape, area_for_mid.shape

    # 2. total area covered by canyon rim
    # this could change if I specify it to be between head/break
    area_canyon = area_for_mid
    total_area_canyon = np.sum(area_canyon)

    # 3. half the area of the canyon
    half_area_canyon = total_area_canyon / 2

    area_canyon.shape, total_area_canyon

    # 4. area at all depths and all ys
    area_all = np.zeros(area_canyon.shape[-3] * area_canyon.shape[-2])
    n = 0
    for depth_ind in range(area_canyon.shape[-3]):
        for y_ind in range(area_canyon.shape[-2]):
            area_all[n] = np.sum(area_canyon[depth_ind, y_ind, :])
            n += 1 

    # 5. relative y_ind where cumulative area is closest to the half area
    cumsum_area_canyon = np.cumsum(area_all)
    relative_ind_half = (np.abs(cumsum_area_canyon-half_area_canyon)).argmin()

    # 6. the real y_ind for half area
    coord_ind_half = relative_ind_half / area_canyon.shape[-2]
    z_ind_half = int(coord_ind_half)
    ind_half = relative_ind_half - (z_ind_half * area_canyon.shape[-2])
    
    return ind_half, area_k, area_k0, area_canyon, area_all, cumsum_area_canyon

# ------------------------------------------------------------------------------------------------

def get_indices_V(gdepw, tmask, tmaskn, mbathy, mbathyn, e1t, e2t):
    
    surface_mask = get_surface_mask(tmaskn, mbathyn)
    
    axis_thalweg = get_index_axis(mbathy, mbathyn)
    
    ind_break = get_index_break(gdepw, mbathyn)
    
    ind_head, ind_shelf = get_index_head(gdepw, tmask, axis_thalweg)
    
    ind_half, area_k, area_k0, area_canyon, area_all, cumsum_area_canyon = get_index_half(tmask, e1t, e2t, surface_mask)
    
    return surface_mask, axis_thalweg, ind_break, ind_head, ind_shelf, ind_half, area_k, area_k0

# ------------------------------------------------------------------------------------------------