import netCDF4 as nc
import numpy as np
import os
import sys
sys.path.append('/ocean/imachuca/Canyons/analysis_mackenzie_canyon/notebooks/general_circulation/')
import general_functions
from salishsea_tools.nc_tools import scDataset


def get_variables_planes(dirname, filename):
    with nc.Dataset(os.path.join(dirname, filename), 'r') as nbl:
        x, y =  slice(1,-1,None), slice(1,-1,None)
        gdepv = nbl.variables['gdepv'][0, :, 1, 1]
        vmask = nbl.variables['vmask'][0, :, y, x]
        mbathy = nbl.variables['mbathy'][0, y, x]
        e1v = nbl.variables['e1v'][0, y, x]
        e3v_0 = nbl.variables['e3v_0'][0, :, y, x]  
        return gdepv, vmask, mbathy, e1v, e3v_0
    
# ------------------------------------------------------------------------------------------------
    
def get_indices_V(gdepv, vmask, mbathy, e1v, e3v_0):
    
    # z index of the shelf platform
    # 1. find where the depth of v point is closest to 80
    # 2. re-assess answer later
    ind_shelf = np.argmin(np.abs(gdepv - 80))

    # y index of shelf break 
    # 1. get top view of vmask at shelf depth
    # 2. extract wet/dry values along x=1 
    # 3. find the first wet cell
    ind_plane = np.where(vmask[ind_shelf, :, 1] == 1)[0][0]

    print('ind_plane', ind_plane)

    # z index of the shelf platform
    cells_shelf = vmask[ind_shelf, ind_plane, :]
    cells_shelf_W = np.count_nonzero(cells_shelf)
    cells_shelf_D = vmask.shape[-1] - cells_shelf_W
    while cells_shelf_D < 4:
        ind_shelf += 1
        cells_shelf = vmask[ind_shelf, ind_plane, :]
        cells_shelf_W = np.count_nonzero(cells_shelf)
        cells_shelf_D = vmask.shape[-1] - cells_shelf_W
    depth_shelf = gdepv[ind_shelf]

    print('ind_shelf', ind_shelf)
    print('depth_shelf', depth_shelf)

    # z index of canyon bottom
    # 1. mbathy gives maximum depth level everywhere
    # 2. this value is given in fortran indexing
    # 3. subtracting 1 gives the deepest level with wet cells
    ind_bottom = (mbathy[ind_plane,:].max())-1
    depth_bottom = gdepv[ind_bottom]

    print('ind_bottom', ind_bottom)
    print('depth_bottom', depth_bottom)

    # x index of canyon axis
    # 1. find all wet cells along canyon bottom
    # 2. find the middle wet cell for symmetric axis
    # 3. this could have a 0.5 so return integer

    ind_axis = int(np.median(np.where(vmask[ind_bottom, ind_plane, :]==1)))
    print('ind_axis', ind_axis)

    # x index of rims
    # 1. last land value on left
    # 2. first land value on right
    ind_rimW0 = np.where(vmask[ind_shelf, ind_plane, :ind_axis]==0)[0][-1]
    ind_rimE0 = np.where(vmask[ind_shelf, ind_plane, ind_axis:]==0)[0][0] + ind_axis
    axis_to_rim = min(ind_rimE0 - ind_axis, ind_axis - ind_rimW0)
    ind_rimW = ind_axis - axis_to_rim
    ind_rimE = ind_axis + axis_to_rim

    print('ind_rimW', ind_rimW)
    print('ind_rimE', ind_rimE)

    # z index of half canyon
    # 1. area of individual wet cells
    cell_x_j = e1v[ind_plane, :]
    cell_y_j = e3v_0[:, ind_plane, :]
    area_j = vmask[:, ind_plane, :] * cell_x_j * cell_y_j

    # 2. total area of every depth level
    area_all = np.zeros(area_j.shape[0])
    for depth_ind in range(area_j.shape[0]):
        area_all[depth_ind] = area_j[depth_ind, :].sum()

    # 3. areas and indices of only the depth levels inside the canyon
    area_canyon = area_all[ind_shelf:ind_bottom]
    area_canyon_inds = np.arange(ind_shelf,ind_bottom)

    # 4. half the area of the canyon
    total_area_canyon = np.sum(area_canyon)
    half_area_canyon = total_area_canyon / 2

    # 5. relative level where cumulative area is closest to the half area
    cumsum_area_canyon = np.cumsum(area_canyon)
    relative_ind_half = (np.abs(cumsum_area_canyon-half_area_canyon)).argmin()

    # 6. the real depth level for half area
    ind_half = area_canyon_inds[relative_ind_half]
    depth_half = gdepv[ind_half]

    print('ind_half', ind_half)
    print('depth_half', depth_half)
    
    # x index of shelves
    # 1. try to make shf sections same width as rim sections
    # 2. if 4 cells away from edge, bring closer to centre
    # 3. find smallest distance from centre and use as width
    ind_shfW0 = ind_rimW - axis_to_rim
    ind_shfE0 = ind_rimE + axis_to_rim
    while ind_shfW0 <= 4:
        ind_shfW0 += 1
    while ind_shfE0 >= vmask.shape[-1]-4:
        ind_shfE0 -= 1
    axis_to_shf = min(ind_shfE0 - ind_axis, ind_axis - ind_shfW0)
    ind_shfW = ind_axis - axis_to_shf
    ind_shfE = ind_axis + axis_to_shf

    print('ind_shfW', ind_shfW)
    print('ind_shfE', ind_shfE)
    
    return ind_plane, ind_shelf, ind_bottom, ind_axis, ind_rimW, ind_rimE, ind_half, ind_shfW, ind_shfE,\
            depth_shelf, depth_bottom, depth_half, area_j
    
# ------------------------------------------------------------------------------------------------

def extract_sections(variable, ind_shelf, ind_bottom, ind_axis, ind_rimW, ind_rimE, ind_half, ind_shfW, ind_shfE):
    ''' Extracts the values of a given variable for
    all the sections dividing the plane at the shelf break.
    '''
    var_shfW = variable[..., : ind_shelf, ind_shfW : ind_rimW]
    var_rimW = variable[..., : ind_shelf, ind_rimW : ind_axis]
    var_rimE = variable[..., : ind_shelf, ind_axis : ind_rimE]
    var_shfE = variable[..., : ind_shelf, ind_rimE : ind_shfE]
    
    var_topW = variable[..., ind_shelf : ind_half + 1, ind_rimW : ind_axis]
    var_topE = variable[..., ind_shelf : ind_half + 1, ind_axis : ind_rimE]
    
    var_botW = variable[..., ind_half + 1 : ind_bottom, ind_rimW : ind_axis]
    var_botE = variable[..., ind_half + 1 : ind_bottom, ind_axis : ind_rimE]
    
    return var_shfW, var_rimW, var_rimE, var_shfE, var_topW, var_topE, var_botW, var_botE

# ------------------------------------------------------------------------------------------------

def total_sections(axis, var_shfW, var_rimW, var_rimE, var_shfE, var_topW, var_topE, var_botW, var_botE, smooth):
    ''' Finds sum of all values in every section of the shelf break plane.
    axis = None for areas
    axis = (1,2) for fluxes
    '''
    tot_var_shfW = general_functions.smooth(np.sum(var_shfW, axis=axis), smooth)
    tot_var_rimW = general_functions.smooth(np.sum(var_rimW, axis=axis), smooth)
    tot_var_rimE = general_functions.smooth(np.sum(var_rimE, axis=axis), smooth)
    tot_var_shfE = general_functions.smooth(np.sum(var_shfE, axis=axis), smooth)

    tot_var_topW = general_functions.smooth(np.sum(var_topW, axis=axis), smooth)
    tot_var_topE = general_functions.smooth(np.sum(var_topE, axis=axis), smooth)
    tot_var_botW = general_functions.smooth(np.sum(var_botW, axis=axis), smooth)
    tot_var_botE = general_functions.smooth(np.sum(var_botE, axis=axis), smooth)
    
    return tot_var_shfW, tot_var_rimW, tot_var_rimE, tot_var_shfE, tot_var_topW, tot_var_topE, tot_var_botW, tot_var_botE

# ------------------------------------------------------------------------------------------------

def get_variables_fluxes(dirname, filepattern, ind_plane, vmask):
    files = general_functions.get_files(dirname, filepattern, 'grid_V')
    x, y =  slice(1,-1,None), int(ind_plane)
    with scDataset(files) as ds:
        vomecrty0 = ds.variables['vomecrty'][:, :, y, x]
    vmask0 = vmask[:, y, :]
    vmask = np.tile(vmask0, (vomecrty0.shape[0],1, 1))  
    vomecrty = np.ma.array(vomecrty0, mask=1 - vmask)
    return vomecrty

# ------------------------------------------------------------------------------------------------

def calculate_flux_V(time_ind, velocity_plane, area_plane):
    '''Calculates flux at all cells at one time.
    velocity_plane (t, z, x)
    area_plane(z, x)
    '''
    Q_plane = velocity_plane[time_ind, :, :] * area_plane
    return Q_plane

# ------------------------------------------------------------------------------------------------

def calculate_flux_V_evolution(velocity_plane, area_plane):
    '''Calculates flux at all cells at all times.
    velocity_plane (t, z, x)
    area_plane(z, x)
    '''
    Q_plane_all = np.zeros_like(velocity_plane)
    for time_ind in range(Q_plane_all.shape[0]):
        Q_plane_all[time_ind, :, :] = calculate_flux_V(time_ind, velocity_plane, area_plane)
    return Q_plane_all