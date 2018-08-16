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
    filesT = general_functions.get_files(dirname, fname, 'grid_T')
    print('files')
    
    y,x = slice(1,-1,None), slice(1,-1,None)

    with scDataset(filesU) as dsU, scDataset(filesV) as dsV, scDataset(filesW) as dsW, scDataset(filesT) as dsT:
        vozocrtx0 = dsU.variables['vozocrtx'][:,dep_ind,y,x]
        print('U')
        vomecrty0 = dsV.variables['vomecrty'][:,dep_ind,y,x]
        print('V')
        vovecrtz0 = dsW.variables['vovecrtz'][:,dep_ind,y,x]
        print('W')
        vosaline0 = dsT.variables['vosaline'][:, dep_ind, y, x]
        vosaline0_orig = dsT.variables['vosaline'][0, dep_ind, y, x]
        print('S')
        sozotaux = dsU.variables['sozotaux'][:,0,0]
        depthu = dsU.variables['depthu'][:]

    with nc.Dataset(os.path.join(dirname, '1_mesh_mask.nc'), 'r') as dsM:
        umask0 = dsM.variables['umask'][0,dep_ind,y,x]
        vmask0 = dsM.variables['vmask'][0,dep_ind,y,x]
        tmask0 = dsM.variables['tmask'][0,dep_ind,y,x]

    umask = np.tile(umask0, (len(sozotaux), 1, 1))
    vmask = np.tile(vmask0, (len(sozotaux), 1, 1))
    tmask = np.tile(tmask0, (len(sozotaux), 1, 1))
    tmask_orig = np.tile(tmask0, (1, 1, 1))  

    vozocrtx = np.ma.array(vozocrtx0, mask=1 - umask)
    vomecrty = np.ma.array(vomecrty0, mask=1 - vmask)
    vovecrtz = np.ma.array(vovecrtz0, mask=1 - tmask)
    vosaline = np.ma.array(vosaline0, mask=1 - tmask)
    vosaline_orig = np.ma.array(vosaline0_orig, mask=1 - tmask_orig)
    
    return vozocrtx, vomecrty, vovecrtz, vosaline, vosaline_orig, umask, vmask, tmask, depthu, sozotaux

def get_vars_for_box(dirname, fname, meshmaskname, x_start, x_end, y_start, y_end, flag):
    if flag=='U':
        gridname = 'grid_U'
        velname = 'vozocrtx'
        depthname = 'depthu'
        maskname = 'umask'
        e1name = 'e1u'
        e2name = 'e2u'
        
    elif flag=='V':
        gridname = 'grid_V'
        velname = 'vomecrty'
        depthname = 'depthv'
        maskname = 'vmask'
        e1name = 'e1v'
        e2name = 'e2v'
        
    elif flag=='W':
        gridname = 'grid_W'
        velname = 'vovecrtz'
        depthname = 'depthw'
        maskname = 'tmask'
        e1name = 'e1t'
        e2name = 'e2t'
    
    filesU = general_functions.get_files(dirname, fname, 'grid_U')
    with scDataset(filesU) as dsU:
        sozotaux = dsU.variables['sozotaux'][:,0,0]
      
    x,y = slice(x_start, x_end, None), slice(y_start, y_end, None)
    files = general_functions.get_files(dirname, fname, gridname) 
    with scDataset(files) as ds:
        vel0 = ds.variables[velname][:,:,y,x]
        depth = ds.variables[depthname][:]

    with nc.Dataset(os.path.join(dirname, meshmaskname), 'r') as dsM:
        mask0 = dsM.variables[maskname][0,:,y,x]
        mask_all = dsM.variables[maskname][0,:,:,:]
        e1 = dsM.variables[e1name][0, y, x]
        e2 = dsM.variables[e2name][0, y, x]

    mask = np.tile(mask0, (len(sozotaux),1, 1, 1))
    vel = np.ma.array(vel0, mask=1 - mask)
    
    return vel, mask, mask_all, depth, e1, e2, sozotaux

def get_sal_cross_mer(dirname, fname, x_ind, time_ind, z_cut):
    
    filesT = general_functions.get_files(dirname, fname, 'grid_T')
    
    y = slice(1,-1,None)
            
    with scDataset(filesT) as dsT:
        vosaline0 = dsT.variables['vosaline'][time_ind, :z_cut, y, x_ind]
        deptht_cm = dsT.variables['deptht'][:z_cut]
        
    with nc.Dataset(os.path.join(dirname, '1_mesh_mask.nc'), 'r') as dsM:
        tmask0 = dsM.variables['tmask'][0, :z_cut, y, x_ind]
        
    tmask_cm = np.tile(tmask0, (1, 1, 1))  
    vosaline_cm = np.ma.array(vosaline0, mask=1 - tmask_cm)
    
    return vosaline_cm, tmask_cm, deptht_cm

def get_sal_cross_zon(dirname, fname, y_ind, time_ind, z_cut):
    
    filesT = general_functions.get_files(dirname, fname, 'grid_T')
    
    x = slice(1,-1,None)
        
    with scDataset(filesT) as dsT:
        vosaline0 = dsT.variables['vosaline'][time_ind, :z_cut, y_ind, x]
        deptht_cz = dsT.variables['deptht'][:z_cut]
        
    with nc.Dataset(os.path.join(dirname, '1_mesh_mask.nc'), 'r') as dsM:
        tmask0 = dsM.variables['tmask'][0, :z_cut, y_ind, x]
        
    tmask_cz = np.tile(tmask0, (1, 1, 1))  
    vosaline_cz = np.ma.array(vosaline0, mask=1 - tmask_cz)
    
    return vosaline_cz, tmask_cz, deptht_cz

# --------------------------------------------------------------------------------------

def get_uv_at_depth(dirname, fname, dep_ind):
    
    filesU = general_functions.get_files(dirname, fname, 'grid_U')        
    filesV = general_functions.get_files(dirname, fname, 'grid_V')
    
    y,x = slice(1,-1,None), slice(1,-1,None)

    with scDataset(filesU) as dsU, scDataset(filesV) as dsV:
        vozocrtx0 = dsU.variables['vozocrtx'][:,dep_ind,y,x]
        vomecrty0 = dsV.variables['vomecrty'][:,dep_ind,y,x]
        sozotaux = dsU.variables['sozotaux'][:,0,0]
        depthu = dsU.variables['depthu'][:]
        depthv = dsV.variables['depthv'][:]

    with nc.Dataset(os.path.join(dirname, '1_mesh_mask.nc'), 'r') as dsM:
        umask0 = dsM.variables['umask'][0,dep_ind,y,x]
        vmask0 = dsM.variables['vmask'][0,dep_ind,y,x]
        e1u = dsM.variables['e1u'][0, y, x]
        e1v = dsM.variables['e1v'][0, y, x]
        e2u = dsM.variables['e2u'][0, y, x]

    umask = np.tile(umask0, (len(sozotaux), 1, 1))
    vmask = np.tile(vmask0, (len(sozotaux), 1, 1))

    vozocrtx = np.ma.array(vozocrtx0, mask=1 - umask)
    vomecrty = np.ma.array(vomecrty0, mask=1 - vmask)
    
    return vozocrtx, vomecrty, umask[0, ...], vmask[0, ...], e1u, e1v, e2u, depthu, depthv, sozotaux


def get_uv_at_depth_day(dirname, fname, dep_ind, day):
    ''' day 1 = 0 to 24 hrs -- time_ind 24 hrs
        day 2 = 24 to 48 hrs -- time_ind 48 hrs
        day 3 = 48 to 72 hrs -- time_ind 72 hrs
        
        enter the actual day you want (no day 0).
    '''
    
    filesU = general_functions.get_files(dirname, fname, 'grid_U')        
    filesV = general_functions.get_files(dirname, fname, 'grid_V')
    
    y,x = slice(1,-1,None), slice(1,-1,None)
    time_ind = day*24

    with scDataset(filesU) as dsU, scDataset(filesV) as dsV:
        vozocrtx0 = dsU.variables['vozocrtx'][time_ind-24:time_ind,dep_ind,y,x]
        vomecrty0 = dsV.variables['vomecrty'][time_ind-24:time_ind,dep_ind,y,x]
        sozotaux = dsU.variables['sozotaux'][time_ind-24:time_ind,0,0]
        depthu = dsU.variables['depthu'][:]
        depthv = dsV.variables['depthv'][:]

    with nc.Dataset(os.path.join(dirname, '1_mesh_mask.nc'), 'r') as dsM:
        umask0 = dsM.variables['umask'][0,dep_ind,y,x]
        vmask0 = dsM.variables['vmask'][0,dep_ind,y,x]
        e1u = dsM.variables['e1u'][0, y, x]
        e2u = dsM.variables['e2u'][0, y, x]
        
    umask = np.tile(umask0, (len(sozotaux), 1, 1))
    vmask = np.tile(vmask0, (len(sozotaux), 1, 1))

    vozocrtx = np.ma.array(vozocrtx0, mask=1 - umask)
    vomecrty = np.ma.array(vomecrty0, mask=1 - vmask)
    
    vozocrtx_avg = np.mean(vozocrtx, axis=0)
    vomecrty_avg = np.mean(vomecrty, axis=0)
    
    return vozocrtx, vomecrty, vozocrtx_avg, vomecrty_avg, umask[0, ...], vmask[0, ...], e1u, e2u, depthu, depthv, sozotaux


def get_1day_avg(vel, day_start, day_end):
    if day_start == None or day_end == None:
        vel_day = np.mean(vel, axis=0)
    else:
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
    u_nstg = u_nstg0[..., ::arrow,::arrow]
    v_nstg = v_nstg0[..., ::arrow,::arrow]
    x_slice = np.arange(1, ugrid.shape[-1])[::arrow] #changed from 1
    y_slice = np.arange(1, ugrid.shape[-2])[::arrow] #changed from 0
    speeds = np.sqrt(u_nstg**2 + v_nstg**2)
    return x_slice, y_slice, u_nstg, v_nstg, speeds

def get_1day_avg_of_speeds(speeds, days):
    ''' Other functions used to set up this process:
    quicklook.get_uv_at_depth
    quicklook.get_speeds
    '''
    speeds_daily = np.full([days, speeds.shape[-2], speeds.shape[-1]], np.nan)
    for d in range(days):
        speeds_daily[d, :, :] = get_1day_avg(speeds, d, d+1) 
    return speeds_daily

def get_sozotaux(dirname, fname):
    filesU = general_functions.get_files(dirname, fname, 'grid_U')
    with scDataset(filesU) as dsU:
        sozotaux = dsU.variables['sozotaux'][:,0,0]
    return sozotaux

def calculate_avg_vel(vel, dep_start, dep_end):

    # find avg U for every row in the y direction
    avg_all_ys = np.mean(np.mean(vel, axis=-1), axis=-2)

    # find avg U within horizontal rectangle for all depths (time, z)
    avg_all_depths = np.mean(np.mean(vel, axis=-1), axis=-1)

    # find avg U for every depth within the box
    avg_box_depths = avg_all_depths[:, dep_start : dep_end+1]

    # find the absolute avg U within the box
    avg_absolute = np.mean(avg_box_depths, axis=-1)
    
    return avg_all_ys, avg_all_depths, avg_box_depths, avg_absolute

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

def plot_incoming_velocity(umask_all, y_start, y_end, x_start, x_end, dep_start, dep_end, avgU_absolute, sozotaux):
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

def plot_vel_sal_story(time_ind, depth, dep_ind, x_ind, y_ind, vozocrtx, vomecrty, vovecrtz, salt_anom, vosaline_cm, vosaline_cz, deptht_cm, deptht_cz):

    cmap = plt.get_cmap(cm.RdBu_r)
    cmap.set_bad('wheat')

    fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(2, 3, figsize=(20, 16))

    levelsU = np.arange(-0.5, 0.6, 0.1)
    levelsV = np.arange(-0.5, 0.6, 0.1)
    levelsW = np.arange(-0.001, 0.0015, 0.0005)
    levelsSA = np.arange(-2, 2.5, 0.5)

    axes = [ax1, ax2, ax3, ax4]
    values = [vozocrtx, vomecrty, vovecrtz, salt_anom]
    levels_all = [levelsU, levelsV, levelsW, levelsSA]
    ttls = ['u velocity [m s$^{-1}$]', 'v velocity [m s$^{-1}$]', 'w velocity [m s$^{-1}$]', 'salinity anomaly [g kg$^{-1}$]']

    for ax, value, levels, ttl, n in zip(axes, values, levels_all, ttls, np.arange(4)):
        ax, xs, ys = general_functions.set_xy(ax, value, 'childm')
        P = ax.pcolormesh(xs, ys, value[time_ind, :, :], vmin = levels[0], vmax = levels[-1], cmap=cmap)
        cs = ax.contour(xs, ys, value[time_ind, :, :], levels = levels, colors='k')
        ax.clabel(cs, inline=1, fontsize=10)
        ax.set_aspect(aspect='equal')
        ax.set_title(ttl, fontsize=20)
        ax.set_ylabel('Cross-shore Distance [m]', fontsize=13)
        ax.set_xlabel('Alongshore Distance [m]', fontsize=13)
        fig.colorbar(P, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03)

    levels = np.arange(21,35, 0.2)
    cmap = plt.get_cmap(cmo.cm.phase)

    c5 = ax5.contourf(ys, deptht_cm, vosaline_cm, levels = levels, cmap = cmap)
    ax5.contour(ys, deptht_cm, vosaline_cm, levels = levels, colors = 'k', alpha=0.8)
    ax5.set_xlim([0, 100000])
    ax5.set_ylim([deptht_cm[-1], 0])
    ax5.set_aspect(aspect=1000)
    ax5.set_xlabel('Cross-shore Distance [m]', fontsize=13)

    c6 = ax6.contourf(xs, deptht_cz, vosaline_cz, levels = levels, cmap = cmap)
    ax6.contour(xs, deptht_cz, vosaline_cz, levels = levels, colors = 'k', alpha=0.8)
    ax6.set_xlim([0, xs[-1]])
    ax6.set_ylim([deptht_cz[-1], 0])
    ax6.set_aspect(aspect=1600)
    ax6.set_xlabel('Alongshore Distance [m]', fontsize=13)

    for ax, c, n in zip([ax5, ax6], [c5, c6], np.arange(2)):
        ax.axhline(depth[dep_ind], c='w', lw=2)
        cbar = plt.colorbar(c, ax=ax, fraction=0.05, pad=0.02, orientation='vertical')
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=0, fontsize=12)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        ax.set_ylabel('Depth [m]', fontsize=13)

    ax4.axhline(ys[y_ind], c='k', lw=1.5)
    ax4.axvline(xs[x_ind], c='k', lw=1.5)

    ax5.set_title('Salinity cross-section in Y direction', fontsize=20)
    ax6.set_title('Salinity cross-section in X direction', fontsize=20)

    plt.tight_layout(h_pad=1.4, w_pad=0.9)
    fig.suptitle('Results at depth = '+str(int(depth[dep_ind]))+' m and time = '+str(time_ind) + ' hrs', fontsize=24)
    plt.subplots_adjust(top=0.95)
    
    return fig

