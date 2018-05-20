# for upwelling
import netCDF4 as nc
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

import sys
sys.path.append('/ocean/imachuca/Canyons/analysis_mackenzie_canyon/notebooks/general_circulation/')
import general_functions
import quicklook

# --------------------------------------------------------------------------------------------

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        resdat = np.asarray(result.data)
        result = np.ma.array(resdat, mask=result.mask, copy=False)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        res = np.interp(result, x, y)
        result = np.ma.array(res, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result
    
# --------------------------------------------------------------------------------------------

def get_vars_salt(dirname, fname, meshmaskname, dep_ind_slice, time_s, time_f, y_end):
           
    if y_end == None:
        y_end = -1
    else:
        y_end = y_end
    
    y,x = slice(1,y_end,None), slice(1,-1,None)
    
    filesT = general_functions.get_files(dirname, fname, 'grid_T')
    filesU = general_functions.get_files(dirname, fname, 'grid_U')
            
    with scDataset(filesT) as dsT, scDataset(filesU) as dsU:
        deptht = dsT.variables['deptht'][:]
        vosaline0 = dsT.variables['vosaline'][time_s:time_f, dep_ind_slice, y, x]
        sozotaux = dsU.variables['sozotaux'][time_s:time_f, 0, 0] 
        
    with nc.Dataset(os.path.join(dirname, meshmaskname), 'r') as dsM:
        tmask0 = dsM.variables['tmask'][0, dep_ind_slice, y, x]
        
    tmask = np.tile(tmask0, (len(sozotaux), 1, 1, 1))  
    vosaline = np.ma.array(vosaline0, mask=1 - tmask)
    
    return vosaline, sozotaux, deptht, tmask[0, 0, ...]

# --------------------------------------------------------------------------------------------

def get_daily_vosaline(vosaline, tmask, days):
    
    # get daily salinity averages from hourly results
    vosaline_daily0 = quicklook.get_1day_avg_of_speeds(vosaline, days)
    tmask_new = np.tile(tmask, (vosaline_daily0.shape[0], 1, 1))
    vosaline_daily = np.ma.array(vosaline_daily0, mask=1 - tmask_new)
    
    return vosaline_daily

# --------------------------------------------------------------------------------------------

def get_daily_depth_ind(vosaline_daily, vosaline_ref_profile):
    
    depth_ind_daily = np.full_like(vosaline_daily, np.nan, dtype=np.int)
    
    # trace back the deptht_ind where this_vosaline occured the vosaline_ref  
    if len(vosaline_daily.shape) == 3:
        for d in range(depth_ind_daily.shape[-3]):
            for y in range(depth_ind_daily.shape[-2]):
                for x in range(depth_ind_daily.shape[-1]):
                    this_vosaline = vosaline_daily[d, y, x]
                    depth_ind_daily[d, y, x] = min(range(len(vosaline_ref_profile)), key=lambda i: abs(vosaline_ref_profile[i]-this_vosaline))
            print(d)
            
    elif len(vosaline_daily.shape) == 1:  
        for d in range(len(depth_ind_daily)):
            this_vosaline = vosaline_daily[d]
            depth_ind_daily[d] = min(range(len(vosaline_ref_profile)), key=lambda i: abs(vosaline_ref_profile[i]-this_vosaline))

    return depth_ind_daily

# --------------------------------------------------------------------------------------------

def get_daily_depth_m(deptht, dep_ind_slice, depth_ind_daily, tmask):

    # get deptht for deptht_ind of origin
    depth_m_daily = deptht[depth_ind_daily]
    # get upwelling displacement by finding difference
    depth_upwelled = depth_m_daily - deptht[dep_ind_slice]
    
    if len(depth_ind_daily.shape) == 3:
        tmask_new = np.tile(tmask, (depth_ind_daily.shape[0], 1, 1))
        depth_m_daily = np.ma.array(depth_m_daily, mask=1 - tmask_new)
        depth_upwelled = np.ma.array(depth_upwelled, mask=1 - tmask_new)
    else:
        pass
    
    return depth_m_daily, depth_upwelled

# --------------------------------------------------------------------------------------------