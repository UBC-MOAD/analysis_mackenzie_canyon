# quick look at cases using plots used in previous notebooks
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean as cmo
import os,sys,fnmatch,time
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from salishsea_tools.nc_tools import scDataset
from matplotlib import ticker
from matplotlib import colors
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.append('/ocean/imachuca/Canyons/analysis_mackenzie_canyon/notebooks/general_circulation/')
import general_functions

# --------------------------------------------------------------------------------------

def get_vars_at_depth(dirname, fname, dep_ind):
    
    filesU = general_functions.get_files(dirname, fname, 'grid_U')        
    filesV = general_functions.get_files(dirname, fname, 'grid_V')
    filesW = general_functions.get_files(dirname, fname, 'grid_W')
    print('files')
    
    y,x = slice(1,-1,None), slice(1,-1,None)

    with scDataset(filesU) as dsU, scDataset(filesV) as dsV, scDataset(filesW) as dsW:
        vozocrtx0 = dsU.variables['vozocrtx'][:,dep_ind,y,x]
        print('U')
        vomecrty0 = dsV.variables['vomecrty'][:,dep_ind,y,x]
        print('V')
        vovecrtz0 = dsW.variables['vovecrtz'][:,dep_ind,y,x]
        print('W')
        sozotaux = dsU.variables['sozotaux'][:,0,0]
        depthu = dsU.variables['depthu'][:]

    with nc.Dataset(os.path.join(dirname, '1_mesh_mask.nc'), 'r') as dsM:
        umask0 = dsM.variables['umask'][0,dep_ind,y,x]
        vmask0 = dsM.variables['vmask'][0,dep_ind,y,x]
        tmask0 = dsM.variables['tmask'][0,dep_ind,y,x]

    umask = np.tile(umask0, (len(sozotaux), 1, 1))
    vmask = np.tile(vmask0, (len(sozotaux), 1, 1))
    tmask = np.tile(tmask0, (len(sozotaux), 1, 1))

    vozocrtx = np.ma.array(vozocrtx0, mask=1 - umask)
    vomecrty = np.ma.array(vomecrty0, mask=1 - vmask)
    vovecrtz = np.ma.array(vovecrtz0, mask=1 - tmask)
    
    return vozocrtx, vomecrty, vovecrtz, umask, vmask, tmask, depthu, sozotaux

def get_vars_for_box(dirname, fname, x_start, x_end, y_start, y_end):
    
    filesU = general_functions.get_files(dirname, fname, 'grid_U')
    
    x,y = slice(x_start, x_end, None), slice(y_start, y_end, None)
    
    with scDataset(filesU) as dsU:
        vozocrtx0 = dsU.variables['vozocrtx'][:,:,y,x]
        sozotaux = dsU.variables['sozotaux'][:,0,0]
        depthu = dsU.variables['depthu'][:]

    with nc.Dataset(os.path.join(dirname, '1_mesh_mask.nc'), 'r') as dsM:
        umask0 = dsM.variables['umask'][0,:,y,x]
        umask_all = dsM.variables['umask'][0,:,:,:]

    umask = np.tile(umask0, (len(sozotaux),1, 1, 1))

    vozocrtx = np.ma.array(vozocrtx0, mask=1 - umask)
    
    return vozocrtx, umask, umask_all, depthu, sozotaux

# --------------------------------------------------------------------------------------

def get_1day_avg(vel, day_start, day_end):
    day = slice(day_start*24, day_end*24, None)
    vel_day = np.mean(vel[day, ...], axis=0)
    return vel_day

def get_speeds(U_vel, V_vel, arrow):
    '''This function unstaggers the velocity components
    and calculates the speeds at arrow intervals.
    
    #from /ocean/imachuca/Canyons/mackenzie_canyon/tools/functions_velocity.py
    '''
    ugrid = U_vel[:]
    vgrid = V_vel[:]
    u_nstg0 = (np.add(ugrid[..., :-1], ugrid[..., 1:]) / 2)[..., 1:, :]
    v_nstg0 = (np.add(vgrid[..., :-1, :], vgrid[..., 1:, :]) / 2)[..., 1:]
    u_nstg = u_nstg0[::arrow,::arrow]
    v_nstg = v_nstg0[::arrow,::arrow]
    x_slice = np.arange(1, ugrid.shape[1])[::arrow]
    y_slice = np.arange(1, ugrid.shape[0])[::arrow]
    speeds = np.sqrt(u_nstg**2 + v_nstg**2)
    return x_slice, y_slice, u_nstg, v_nstg, speeds

def calculate_avgU(vozocrtx, dep_start, dep_end):

    # find avg U for every row in the y direction
    avgU_all_ys = np.mean(np.mean(vozocrtx, axis=-1), axis=-2)

    # find avg U within horizontal rectangle for all depths (time, z)
    avgU_all_depths = np.mean(np.mean(vozocrtx, axis=-1), axis=-1)

    # find avg U for every depth within the box
    avgU_box_depths = avgU_all_depths[:, dep_start : dep_end+1]

    # find the absolute avg U within the box
    avgU_absolute = np.mean(avgU_box_depths, axis=-1)
    
    return avgU_all_ys, avgU_all_depths, avgU_box_depths, avgU_absolute

# --------------------------------------------------------------------------------------

def plot_vel_snapshots_depth(vel_all, depth, dep_ind, vm, ttl):
    cmap = plt.get_cmap(cmo.cm.balance)
    cmap.set_bad('wheat')
    fig, axes = plt.subplots(3, 5, figsize=(20,15), sharey=True)
    for ax, n in zip(axes.flatten(), np.arange(15)):
        vel_dayn = get_1day_avg(vel_all, n, n+1)
        ax, xs, ys = general_functions.set_xy(ax, vel_dayn, 'childm')
        p = ax.pcolormesh(xs, ys, vel_dayn, vmin=-1*vm, vmax=vm, cmap=cmap)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xlim([0, xs[-1]])
        ax.set_ylim([0, ys[-1]])
        ax.set_title('Day '+str(n+1), fontsize=16)
        if n==0:
            fig.colorbar(p, ax=ax)
    fig.tight_layout(w_pad=0.1, h_pad=0.25)
    fig.suptitle(ttl+' Profile at Depth '+str(np.round(depth[dep_ind],1))+' m', fontsize=20)
    plt.subplots_adjust(top=0.92)
    return fig

def plot_speed_streamlines_depth(vozocrtx, vomecrty, umask, depth, dep_ind, vmin, vmax):
    norm = mpl.colors.Normalize(vmin, vmax)
    cmap = LinearSegmentedColormap.from_list('mycmap', ['wheat', 'white'])

    fig, axes = plt.subplots(3, 4, figsize = (20, 20))
    for ax, n in zip(axes.flatten(), np.arange(12)):
        U_vel = get_1day_avg(vozocrtx, n, n+1)
        V_vel = get_1day_avg(vomecrty, n, n+1)
        x_slice, y_slice, u_nstg, v_nstg, speeds = get_speeds(U_vel, V_vel, 1)
        ax.pcolormesh(umask[0,:,:], cmap=cmap, zorder=1)
        strm = ax.streamplot(x_slice, y_slice, u_nstg, v_nstg, 
                             color=speeds, cmap=cmo.cm.matter, density=2, linewidth=2, norm=norm, arrowsize=2, zorder=2)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_aspect(aspect='equal')
        ax.set_xlim([0, umask.shape[-1]])
        ax.set_ylim([0, umask.shape[-2]])
        ax.set_title('Day ' + str(n+1), fontsize=20)
        if n == 0:
            cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.05, orientation='horizontal', pad=0.009)
            cbar.set_label('Speeds [m/s]', fontsize=18, rotation=0, labelpad=2.5)
            cbar.ax.tick_params(labelsize=13)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()
        else:
            pass
        
    fig.tight_layout(w_pad=1.2, h_pad=0.01)
    fig.suptitle('Streamlines Using U and V Velocity Components [Depth = '+str(int(depth[dep_ind]))+' m]', fontsize=25)
    plt.subplots_adjust(top=0.95)
    return fig

def plot_speed_pcolor_depth(vozocrtx, vomecrty, umask, depth, dep_ind, vmin, vmax, levels):
    colour_list = ["#c8274c",
    "#f25546",
    "#F06543",
    "#e96e33",
    "#f0b038",
    "#FFE74C",
    "#69b944",
    "#72b286",
    "#69b0bc",
    "#619ee4",
    "#4b5bbb"][::-1]
    cmap = LinearSegmentedColormap.from_list('mycmap', colour_list, N=500, gamma=1)
    cmap.set_bad('wheat')

    fig, axes = plt.subplots(3, 4, figsize = (20, 20))
    for ax, n in zip(axes.flatten(), np.arange(12)):
        U_vel = get_1day_avg(vozocrtx, n, n+1)
        V_vel = get_1day_avg(vomecrty, n, n+1)
        x_slice, y_slice, u_nstg, v_nstg, speeds = get_speeds(U_vel, V_vel, 1)

        p = ax.pcolormesh(x_slice, y_slice, speeds, cmap=cmap, vmin=vmin, vmax=vmax)
        cs = ax.contour(x_slice, y_slice, speeds, levels = levels, colors='k', alpha=0.5)
        ax.clabel(cs, inline=1, fontsize=10)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_aspect(aspect='equal')
        ax.set_xlim([0, umask.shape[-1]])
        ax.set_ylim([0, umask.shape[-2]])
        ax.set_title('Day ' + str(n+1), fontsize=20)
        if n == 0:
            cbar = fig.colorbar(p, ax=ax, fraction=0.05, orientation='horizontal', pad=0.009)
            cbar.set_label('Speeds [m/s]', fontsize=18, rotation=0, labelpad=2.5)
            cbar.ax.tick_params(labelsize=13)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()
        else:
            pass
        
    fig.tight_layout(w_pad=1.2, h_pad=0.01)
    fig.suptitle('Speeds [Depth = '+str(int(depth[dep_ind]))+' m]', fontsize=25)
    plt.subplots_adjust(top=0.95)
    return fig

def plot_story(umask_all, y_start, y_end, x_start, x_end, dep_start, dep_end, avgU_absolute, sozotaux):
    cmap = LinearSegmentedColormap.from_list('mycmap', ['wheat', 'white'])
    
    convert_to_distance = 2/3 
    xs = np.arange(umask_all.shape[-1]) * convert_to_distance
    ys = np.arange(umask_all.shape[-2]) * convert_to_distance
    zs = np.arange(umask_all.shape[-3])

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])
    ax1 = plt.subplot(gs[0])
    ax4 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax2 = plt.subplot(gs[3])

    s = ax1.pcolormesh(xs, ys, umask_all[9, :,:], cmap=cmap)
    ax1.set_aspect(aspect='equal')
    ax1.add_patch(patches.Rectangle((x_start * convert_to_distance, y_start * convert_to_distance),
                                    (x_end-x_start) * convert_to_distance,(y_end-y_start) * convert_to_distance, 
                                    fill=False, linewidth=3))
    ax1.set_xlim([0, xs[-1]])
    ax1.set_ylim([0, ys[-1]])
    ax1.set_xlabel('Alongshore Distance [km]', fontsize = 15)
    ax1.set_ylabel('Cross-shore Distance [km]', fontsize = 15)

    sc = ax3.pcolormesh(ys, zs, umask_all[:, :, x_start], cmap=cmap)
    ax3.add_patch(patches.Rectangle((y_start * convert_to_distance, dep_start),(y_end-y_start) * convert_to_distance, 
                                    dep_end-dep_start, fill=False, linewidth=3))
    ax3.set_ylim([zs[-1], 0])
    ax3.set_xlim([0, ys[-1]])
    ax3.set_ylabel('Z Indices', fontsize = 15)
    ax3.set_xlabel('Cross-shore Distance [km]', fontsize = 15)

    avgU_absolute_smoothed = general_functions.smooth(avgU_absolute[:], 12)
    ax2.plot(avgU_absolute, c='gray', lw=2, alpha=0.8)
    ax2.plot(avgU_absolute_smoothed, c='k', lw=2)
    ax2.set_title('Incoming Velocity', fontsize=20)
    ax2.set_xlabel('Time [hours]', fontsize=15)
    ax2.set_ylabel('[ms$^{-1}$]', fontsize=20)
    ax2.set_xlim([0, 480])
    ax2.set_ylim([-0.3, 0.3])
    ml = MultipleLocator(24)
    ax2.xaxis.set_minor_locator(ml)
    ax2.xaxis.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax2.axhline(0, linestyle='--', c='gray')

    ax4.plot(sozotaux, c='k', lw=2)
    ax4.set_title('Wind Stress', fontsize=20)
    ax4.set_xlabel('Time [hours]', fontsize=15)
    ax4.set_ylabel('[Nm$^{-2}$]', fontsize=20)
    ax4.set_xlim([0, 480])
    ax4.set_ylim([-1.0, 1.0])
    ml = MultipleLocator(24)
    ax4.xaxis.set_minor_locator(ml)
    ax4.xaxis.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax4.axhline(0, linestyle='--', c='gray')

    fig.tight_layout(w_pad=5.5, h_pad=3.5)
    
    print('min unsmoothed: ', avgU_absolute.min())
    print('min smoothed: ', avgU_absolute_smoothed.min())
    return fig

