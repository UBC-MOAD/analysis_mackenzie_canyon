# script with comparisons (wave, nitrate, upwelling) for wind forcing

import netCDF4 as nc
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean as cmo
import os,sys,fnmatch,time
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import animation
from salishsea_tools.nc_tools import scDataset
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib as mpl
import numpy.ma as ma
from scipy import interpolate
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import sys
sys.path.append('/ocean/imachuca/Canyons/analysis_mackenzie_canyon/notebooks/upwelling_depth/')
import upwelling_functions
sys.path.append('/ocean/imachuca/Canyons/analysis_mackenzie_canyon/notebooks/general_circulation/')
import general_functions
import quicklook
sys.path.append('/ocean/imachuca/Canyons/analysis_mackenzie_canyon/notebooks/biology_question/')
import functions_nutrients

# ---------------------------------------------------------------

def run_script(kind, flag):
    print(flag)
    
    dep_inds = np.arange(0, 80)
    
    # ---------------------------------------------------------------
    
    cases = ['half', 'base', 'double']

    comparison_matrix = np.full([len(dep_inds)+1, 4], np.nan)

    for n in range(len(dep_inds)):
        dep_ind = dep_inds[n]
        print(dep_ind)

        if flag == 'wave':
            comparison_list, gdep = compare_wave(kind, cases, dep_ind)
        elif flag == 'nitrate':
            comparison_list, gdep = compare_nitrate(kind, cases, dep_ind)
        elif flag == 'upwelling':
            comparison_list, gdep = compare_upwelling(kind, cases, dep_ind)

        comparison_normalized = comparison_list/comparison_list[1]

        comparison_matrix[n,0] = gdep
        comparison_matrix[n,1] = comparison_normalized[0]
        comparison_matrix[n,2] = comparison_normalized[1]
        comparison_matrix[n,3] = comparison_normalized[2]

    comparison_matrix[-1, 1] = np.nanmean(comparison_matrix[:, 1])
    comparison_matrix[-1, 2] = np.nanmean(comparison_matrix[:, 2])
    comparison_matrix[-1, 3] = np.nanmean(comparison_matrix[:, 3])

    print(comparison_matrix)

    np.savetxt('./files/comparison_'+flag+'_'+kind+'.out', (comparison_matrix), delimiter=',')
    print('done')
    return

# ---------------------------------------------------------------

def get_varbs_wave(kind, case, dep_ind):
        
    if kind == 'ideal':
        x_start =  202
        x_end =  207
        y_start =  164
        y_end =  204
    elif kind == 'real':
        x_start =  180
        x_end =  185
        y_start =  157
        y_end =  197

    # ------------------------------------------------
    
    dirname = '/ocean/imachuca/Canyons/results_mackenzie/extended_domain/'+kind+'_'+case+'/'
    fname = '1_MCKNZ_1h_20170101_201701*'
    meshmaskname = '1_mesh_mask.nc'
    time_end = 6*24
        
    filesW = general_functions.get_files(dirname, fname, 'grid_W')
    
    y,x = slice(y_start, y_end,None), slice(x_start, x_end,None)
    
    with scDataset(filesW) as dsW:
        vovecrtz = dsW.variables['vovecrtz'][:time_end, dep_ind, y, x] * 1000
        
    with nc.Dataset(os.path.join(dirname, meshmaskname), 'r') as dsM:
        gdepw = dsM.variables['gdepw_1d'][0, dep_ind]
        
    return vovecrtz, gdepw

# ---------------------------------------------------------------

def get_varbs_nitrate(kind, case, dep_ind):
    
    dirname = '/ocean/imachuca/Canyons/results_mackenzie/extended_domain/'+kind+'_'+case+'/'
    fname = '1_MCKNZ_1h_20170101_201701*'
    meshmaskname = '1_mesh_mask.nc'
    time_end = 6*24
    
    filesT = general_functions.get_files(dirname, fname, 'grid_T')
    filesW = general_functions.get_files(dirname, fname, 'grid_W')
    
    y, x = slice(1,-1,None), slice(1,-1,None)
    
    with scDataset(filesT) as dsT, scDataset(filesW) as dsW:
        vosaline0 = dsT.variables['vosaline'][:time_end,dep_ind,y,x]
        vovecrtz0 = dsW.variables['vovecrtz'][:time_end,dep_ind,y,x]
        
    with nc.Dataset(os.path.join(dirname, meshmaskname), 'r') as dsM:
        e1t0 = dsM.variables['e1t'][0, y, x] # m
        e2t0 = dsM.variables['e2t'][0, y, x] # m
        gdept = dsM.variables['gdept_1d'][0, dep_ind]
    
    return vosaline0, vovecrtz0, e1t0, e2t0, gdept

# ---------------------------------------------------------------

def get_varbs_upwelling(kind, case, dep_ind):
    
    dirname = '/ocean/imachuca/Canyons/results_mackenzie/extended_domain/'+kind+'_'+case+'/'
    fname = '1_MCKNZ_1h_20170101_201701*'
    meshmaskname = '1_mesh_mask.nc'
    time_end = 6*24
    time_s = 0
    
    vosaline_ref = nc.Dataset('/ocean/imachuca/Canyons/mackenzie_canyon/conditions/NEMO_files/salinity/salinity_for_agrif.nc')['vosaline'][:]
    vosaline_ref_profile = vosaline_ref[0, :, 0, 0]
    
    y,x = slice(1,-1,None), slice(1,-1,None)

    vosaline, sozotaux, deptht, tmask = upwelling_functions.get_vars_salt(dirname, fname, meshmaskname, 
                                                                            dep_ind, time_s, time_end, None)
    
    max_vosaline = np.full([vosaline.shape[0]], np.nan)
    
    for t in range(vosaline.shape[0]):
        max_vosaline[t] = np.nanmax(vosaline[t, :, :])

    max_depth_ind = upwelling_functions.get_daily_depth_ind(max_vosaline, vosaline_ref_profile)
        
    original_depth, max_depth_upwelled = upwelling_functions.get_daily_depth_m(deptht, dep_ind, max_depth_ind, tmask)
        
    return max_depth_upwelled, deptht[dep_ind]

# ---------------------------------------------------------------

def compare_wave(kind, cases, dep_ind):
    
    comparison_list = []
    
    for case, n in zip(cases, np.arange(len(cases))):
        vovecrtz, gdepw = get_varbs_wave(kind, case, dep_ind)      
        comparison = np.nanmean(np.where(vovecrtz>=0, vovecrtz, np.nan)) 
        #comparison_value = np.nanmax(np.where(vovecrtz>=0, vovecrtz, np.nan))
        comparison_list.append(comparison)
        
    return comparison_list, gdepw

# ---------------------------------------------------------------

def compare_nitrate(kind, cases, dep_ind):
    
    file = '/ocean/imachuca/Canyons/analysis_mackenzie_canyon/notebooks/biology_question/LSSL_Geochemistry2009.xls'
    sheet = '2009-20_LSSL_Chem'

    nutrient = 'nitrate'
    NO3_smooth, salt_new = functions_nutrients.get_nutrient_profile(file, sheet, nutrient)
    salt_line, nut_line = salt_new, NO3_smooth
    
    f_interp_final = interpolate.interp1d(salt_line, nut_line, fill_value='extrapolate')
    
    comparison_list = []
    
    for case, n in zip(cases, np.arange(len(cases))):
        vosaline0, vovecrtz0, e1t0, e2t0, gdept = get_varbs_nitrate(kind, case, dep_ind)
        nut_plan = np.full_like(vosaline0, np.nan)
        nut_flux = np.full_like(vosaline0, np.nan)
        
        for t in range(vosaline0.shape[-3]):
            for j in range(vosaline0.shape[-2]):
                for i in range(vosaline0.shape[-1]):
                    nut_plan[t, j, i] = f_interp_final(vosaline0[t, j, i])
            nut_flux[t, :, :] = nut_plan[t, :, :] * vovecrtz0[t, :, :] * e1t0 * e2t0
        
        total_flux = np.nansum(np.nansum(nut_flux, axis=1), axis=1)
        comparison = np.nansum(total_flux)
        
        comparison_list.append(comparison)
        
    return comparison_list, gdept

# ---------------------------------------------------------------

def compare_upwelling(kind, cases, dep_ind):
    
    comparison_list = []
    
    for case, n in zip(cases, np.arange(len(cases))):
        max_depth_upwelled, gdep = get_varbs_upwelling(kind, case, dep_ind)      
        comparison = np.nanmax(max_depth_upwelled) 
        comparison_list.append(comparison)
        
    return comparison_list, gdep

# ---------------------------------------------------------------

#run_script('ideal', 'wave')
#print('done')
#run_script('ideal', 'nitrate')
#print('done')
run_script('ideal', 'upwelling')
print('done')

# ---------------------------------------------------------------
