import netCDF4 as nc
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean as cmo
import os,sys,fnmatch,time
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import datetime
import pandas as pd
import math
from scipy import interpolate

# =========================================================================

def get_nutrient_profile(file, sheet, nutrient):
    df_data = pd.read_excel(file, sheetname=sheet)

    if nutrient == 'phosphate':
        nutrient_varb = 'PO4\n[mmol/m3]'
    if nutrient == 'nitrate':
        nutrient_varb = 'NO3\n[mmol/m3]'

    print(nutrient)

    nut = np.array(df_data[nutrient_varb])

    salt = np.array(df_data['Salt\n[PSS-78]      '])
    lat = np.array(df_data['LAT DEG'])
    lon = np.array(df_data['LON DEG'])
    time = np.array(df_data['Cast Start Time [UTC]'])

    lat_all = []
    lon_all = []
    time_all = []
    salt_all = []
    nut_all = []
    inds = []

    for i in (np.where(np.logical_and(lat>=70, lat<=71))[0]):
        if np.logical_and(lon[i]>=139, lon[i]<=140):
            inds.append(i)
            lat_all.append(lat[i])
            lon_all.append(lon[i])
            time_all.append(time[i])
            salt_all.append(salt[i])
            nut_all.append(nut[i])
        else:
            pass

    lat_all = np.array(lat_all)
    lon_all = np.array(lon_all)
    time_all = np.array(time_all)
    salt_all = np.array(salt_all)
    nut_all = np.array(nut_all)

    #print(np.unique(lat_all))
    #print(np.unique(lon_all))
    #print(inds)
    #print(time_all)

    nut_smooth, salt_new = smooth_nut(nut_all, salt_all)

    print('lat', lat_all.shape, 'salt', salt_new.shape, 'nut', nut_smooth.shape)

    np.savetxt('../thesis/files/'+nutrient+'_info.out', (lat_all, lon_all, time_all), delimiter=',')

    np.savetxt('../thesis/files/'+nutrient+'_prof.out', (salt_new, nut_smooth), delimiter=',')

    return nut_smooth, salt_new

#==============================================================================

def smooth_nut(nut_all, salt_all):
    idx_finite = np.isfinite(nut_all)
    f_interp = interpolate.interp1d(salt_all[idx_finite], nut_all[idx_finite], fill_value='extrapolate')

    salt_new = np.arange(np.nanmin(salt_all), np.nanmax(salt_all), 0.5)#0.01
    nut_new = f_interp(salt_new)

    nut_smooth = np.copy(nut_new)
    return nut_smooth, salt_new
