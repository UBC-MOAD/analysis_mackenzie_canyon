# Started April, 2018
# Used as shortcuts for big story plots used to answer research questions about circulation
# quicklook.py also has important functions

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
import quicklook

# --------------------------------------------------------------------------------------
# --------------- Calculations
# --------------------------------------------------------------------------------------

def find_separation_velocity(speeds_daily, umask, e1u, days, arrow, model):
    ''' 
    y_break = constrain the analysis to velocities inside the canyon
    speeds_cross = grabs speeds at all x inds for a given y ind
    speeds_cross_max = identifies the highest speed for a given y ind
    speeds_cross_max_xind = finds the x ind where the max occurs,
        therefore completing the coordinate for the highest speed
    water_inds = finds all indices that are wet, if a given y only has
        land (onshore of the head) then the length of the list is 0
    eastmost_water_xind = gives the upstream x ind for the canyon wall
    xinds_separation = number of x indices between the eastmost wet cell
        and the cell where the highest speeds occur
    
    Other functions used to set up this process:
    quicklook.get_uv_at_depth
    quicklook.get_speeds
    quicklook.get_1day_avg_of_speeds
    '''

    if model == 'ideal':
        y_break = np.where(umask[:, 0]==0)[0][-1]
    elif model == 'real':
        pass
    
    if arrow==1:
        umask_slice = umask[:-1, :-1]
    else:
        umask_slice = umask[::arrow, ::arrow]
        
        
    speeds_cross = np.full([days, y_break, speeds_daily.shape[-1]], np.nan)
    speeds_cross_max = np.full([days, y_break], np.nan)
    speeds_cross_max_xind = np.full([days, y_break], np.nan)
    eastmost_water_xind = np.full([y_break], np.nan)
    e1u_mean = np.full([y_break], np.nan)
    xinds_separation = np.full([days, y_break], np.nan)
    dist_separation = np.full([days, y_break], np.nan)
    
    for day in range(days):
        for y in range(y_break):
            speeds_cross[day, y, :] = speeds_daily[day, y, :]
            
            speeds_cross_max_value = np.nanmax(speeds_cross[day, y, :])
            if speeds_cross_max_value != 0:
                speeds_cross_max[day, y] = speeds_cross_max_value
            else:
                pass
            
            speeds_cross_max_xind_value = np.where(speeds_cross[day, y, :]==speeds_cross_max[day, y])[0]
            if len(speeds_cross_max_xind_value) != 0:
                speeds_cross_max_xind[day, y] = speeds_cross_max_xind_value[0]
            else:
                pass
            
            water_inds = np.where(umask_slice[y, :]==1)[0]
            if len(water_inds) != 0:
                eastmost_water_xind[y] = water_inds[-1]
            else:
                pass
            
            e1u_mean[y] = np.mean(e1u[y, :])
        
        xinds_separation[day, :] = eastmost_water_xind - speeds_cross_max_xind[day, :]
        dist_separation[day, :] = xinds_separation[day, :] * e1u_mean
    
    return y_break, speeds_cross, speeds_cross_max, speeds_cross_max_xind, eastmost_water_xind, xinds_separation, dist_separation

# --------------------------------------------------------------------------------------
# --------------- Plots
# --------------------------------------------------------------------------------------

def plot_speed_pcolormesh(x_slice, y_slice, speeds_daily, umask, levels, depthu, dep_ind, arrow, case):
    ''' Other functions used to set up this process:
    quicklook.get_uv_at_depth
    quicklook.get_speeds
    quicklook.get_1day_avg_of_speeds
    '''
    
    colour_list = ["#c8274c","#f25546","#F06543","#e96e33","#f0b038","#FFE74C",
                   "#69b944","#72b286","#69b0bc","#619ee4","#4b5bbb"][::-1]

    cmap = LinearSegmentedColormap.from_list('mycmap', colour_list, N=500, gamma=1)
    cmap.set_bad('silver')
    vmin, vmax = 0, speeds_daily.max()
    if arrow==1:
        umask_slice = umask[:-1, :-1]
    else:
        umask_slice = umask[::arrow, ::arrow]

    fig, axes = plt.subplots(3, 3, figsize = (20, 21))
    for ax, n in zip(axes.flatten(), np.arange(9)):
        plot_speeds = np.ma.array(speeds_daily[n, ...], mask=1 - umask_slice)
        p = ax.pcolormesh(x_slice, y_slice, plot_speeds, cmap=cmap, vmin=vmin, vmax=vmax)
        cs = ax.contour(x_slice, y_slice, plot_speeds, levels = levels, colors='k', alpha=0.5)
        ax.clabel(cs, inline=1, fontsize=10)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_aspect(aspect='equal')
        ax.set_xlim([0, umask.shape[-1]])
        ax.set_ylim([0, umask.shape[-2]])
        ax.set_title('Day ' + str(n+1), fontsize=20)
        cbar = fig.colorbar(p, ax=ax, fraction=0.05, orientation='horizontal', pad=0.009)
        cbar.ax.tick_params(labelsize=13)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

    fig.tight_layout(w_pad=1.2, h_pad=0.01)
    fig.suptitle(case+' - Speeds [m/s] at depth = '+str(int(depthu[dep_ind]))+' m', fontsize=25)
    plt.subplots_adjust(top=0.96)
    return fig

# --------------------------------------------------------------------------------------

def plot_speed_quiver(x_slice, y_slice, u_nstg_daily, v_nstg_daily, speeds_daily, umask, vmask, levels, depthu, dep_ind, arrow, case):
    ''' Other functions used to set up this process:
    quicklook.get_uv_at_depth
    quicklook.get_speeds
    quicklook.get_1day_avg_of_speeds
    '''
    
    cmap = cm.Blues
    cmap.set_bad('silver')
    vmin, vmax = 0, speeds_daily.max()
    if arrow==1:
        umask_slice = umask[:-1, :-1]
        vmask_slice = vmask[:-1, :-1]
    else:
        umask_slice = umask[::arrow, ::arrow]
        vmask_slice = vmask[::arrow, ::arrow]

    fig, axes = plt.subplots(3, 3, figsize = (20, 21))
    for ax, n in zip(axes.flatten(), np.arange(9)):
        plot_speeds = np.ma.array(speeds_daily[n, ...], mask=1 - umask_slice)
        plot_u_nstg = np.ma.array(u_nstg_daily[n, ...], mask=1 - umask_slice)
        plot_v_nstg = np.ma.array(v_nstg_daily[n, ...], mask=1 - vmask_slice)
        p = ax.pcolormesh(x_slice, y_slice, plot_speeds, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.quiver(x_slice, y_slice, plot_u_nstg, plot_v_nstg, color='k', 
                  clim=[vmin,vmax], pivot='mid', width=0.005, headwidth=2.5)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_aspect(aspect='equal')
        ax.set_xlim([0, x_slice[-1]])
        ax.set_ylim([0, y_slice[-1]])
        ax.set_title('Day ' + str(n+1), fontsize=20)
        cbar = fig.colorbar(p, ax=ax, fraction=0.05, orientation='horizontal', pad=0.009)
        cbar.ax.tick_params(labelsize=13)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

    fig.tight_layout(w_pad=1.2, h_pad=0.01)
    fig.suptitle(case+' - Speeds [m/s] at depth = '+str(int(depthu[dep_ind]))+' m', fontsize=25)
    plt.subplots_adjust(top=0.96)
    return fig

# --------------------------------------------------------------------------------------

def plot_separation_path(umask, speeds_cross_max, speeds_cross_max_xind, eastmost_water_xind, y_break, depthu, dep_ind, arrow, case):
    ''' Other functions used to set up this process:
    quicklook.get_uv_at_depth
    quicklook.get_speeds
    quicklook.get_1day_avg_of_speeds
    circulation_plots.find_separation_velocity
    '''
    
    if arrow==1:
        umask_slice = umask[:-1, :-1]
    else:
        umask_slice = umask[::arrow, ::arrow]
        
    colour_list = ["#c8274c","#f25546","#F06543","#e96e33","#f0b038","#FFE74C",
                   "#69b944","#72b286","#69b0bc","#619ee4","#4b5bbb"][::-1]

    cmap = LinearSegmentedColormap.from_list('mycmap', colour_list, N=500, gamma=1)
    cmap_mask = LinearSegmentedColormap.from_list('mycmap', ['silver', 'white'])
    #vmin, vmax = 0, np.nanmax(speeds_cross_max)*0.5
    every = 1
    
    fig, axes = plt.subplots(3, 3, figsize = (20, 21))
    for ax, n in zip(axes.flatten(), np.arange(9)):
        ax.pcolormesh(umask_slice, cmap=cmap_mask)
        ax.plot(speeds_cross_max_xind[n, :], np.arange(y_break), c='k', lw=2, zorder=1)
        p = ax.scatter(speeds_cross_max_xind[n, ::every], np.arange(y_break)[::every], c=speeds_cross_max[n, ::every],
                   cmap=cmap, edgecolors='none', s=70, marker=',', zorder=2)#, vmin=vmin, vmax=vmax
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_aspect(aspect='equal')
        ax.set_xlim([umask_slice.shape[-1]*0.15, umask_slice.shape[-1]*0.85])
        ax.set_ylim([0, umask_slice.shape[-2]*0.7])
        ax.set_title('Day ' + str(n+1), fontsize=20)
        cbar = fig.colorbar(p, ax=ax, fraction=0.05, orientation='horizontal', pad=0.009)
        cbar.ax.tick_params(labelsize=13)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

    fig.tight_layout(w_pad=1.2, h_pad=0.01)
    fig.suptitle(case+' - Highest speeds [m/s] at depth = '+str(int(depthu[dep_ind]))+' m', fontsize=25)
    plt.subplots_adjust(top=0.96)
    return fig

# --------------------------------------------------------------------------------------

def plot_max_speeds_daily(speeds_cross_max, y_break, umask, vmax, depthu, dep_ind, case):

    fig = plt.figure(figsize=(20, 25))
    ax0 = plt.subplot2grid((4, 3), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((4, 3), (1, 0), colspan=1)
    ax2 = plt.subplot2grid((4, 3), (1, 1), rowspan=1)
    ax3 = plt.subplot2grid((4, 3), (1, 2), rowspan=1)
    ax4 = plt.subplot2grid((4, 3), (2, 0), colspan=1)
    ax5 = plt.subplot2grid((4, 3), (2, 1), rowspan=1)
    ax6 = plt.subplot2grid((4, 3), (2, 2), rowspan=1)
    ax7 = plt.subplot2grid((4, 3), (3, 0), colspan=1)
    ax8 = plt.subplot2grid((4, 3), (3, 1), rowspan=1)
    ax9 = plt.subplot2grid((4, 3), (3, 2), rowspan=1)

    ticks=np.linspace(0, (y_break*1.1), 10, dtype=int)
    for ax, n in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 ], np.arange(9)):
        plot_speed = speeds_cross_max[n, :]
        plot_ys = np.arange(y_break)
        max_plot_speed = np.round(np.nanmax(plot_speed),2)
        min_plot_speed = np.round(np.nanmin(plot_speed),2)

        ax.plot(plot_speed, plot_ys, c='orchid', lw=2, zorder=1)
        ax.scatter(plot_speed, plot_ys, c='indigo', zorder=2, s=5)
        ax.set_xlim([0, vmax])
        ax.set_ylim([ticks[0], ticks[-1]])
        ax.set_title('Day ' + str(n+1), fontsize=20)
        ax.yaxis.set_ticks(ticks)
        ax.grid()
        ax.annotate('max = '+str(max_plot_speed), xy=(vmax*0.7, y_break*0.3), fontsize=20)
        ax.annotate('min = '+str(min_plot_speed), xy=(vmax*0.7, y_break*0.2), fontsize=20)

    cmap_mask = LinearSegmentedColormap.from_list('mycmap', ['silver', 'white'])
    ax0.pcolormesh(umask, cmap=cmap_mask)
    ax0.set_xlim([0, umask.shape[-1]])
    ax0.set_ylim([0, umask.shape[-2]])
    ax0.set_aspect(aspect='equal')
    ax0.set_title('Child Domain', fontsize=20)
    ax0.yaxis.set_ticks(ticks)
    ax0.yaxis.grid(color='k')

    fig.tight_layout(w_pad=1.2, h_pad=0.5)
    fig.suptitle(case+' - Highest speeds [m/s] for every y index at depth = '+str(int(depthu[dep_ind]))+' m', fontsize=25)
    plt.subplots_adjust(top=0.95)
    
    return fig

# --------------------------------------------------------------------------------------

def plot_separation_daily(kind_separation, kind_index, y_break, umask, e1u, vmax, depthu, dep_ind, case):

    fig = plt.figure(figsize=(20, 25))
    ax0 = plt.subplot2grid((4, 3), (0, 1), colspan=1)
    ax1 = plt.subplot2grid((4, 3), (1, 0), colspan=1)
    ax2 = plt.subplot2grid((4, 3), (1, 1), rowspan=1)
    ax3 = plt.subplot2grid((4, 3), (1, 2), rowspan=1)
    ax4 = plt.subplot2grid((4, 3), (2, 0), colspan=1)
    ax5 = plt.subplot2grid((4, 3), (2, 1), rowspan=1)
    ax6 = plt.subplot2grid((4, 3), (2, 2), rowspan=1)
    ax7 = plt.subplot2grid((4, 3), (3, 0), colspan=1)
    ax8 = plt.subplot2grid((4, 3), (3, 1), rowspan=1)
    ax9 = plt.subplot2grid((4, 3), (3, 2), rowspan=1)
    
    if kind_index == '[m]' or kind_index == '[indices]':
        divide = 1
    elif kind_index == '[km]':
        divide = 1000

    ticks=np.linspace(0, (y_break*1.1), 10, dtype=int)
    for ax, n in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 ], np.arange(9)):
        plot_separation = kind_separation[n, :] / divide
        plot_ys = np.arange(y_break)
        max_plot_separation = np.round(np.nanmax(plot_separation),2)
        min_plot_separation = np.round(np.nanmin(plot_separation),2)

        ax.plot(plot_separation, plot_ys, c='cornflowerblue', lw=2, zorder=1)
        ax.scatter(plot_separation, plot_ys, c='navy', zorder=2, s=5)
        ax.set_xlim([vmax, 0])
        ax.set_ylim([ticks[0], ticks[-1]])
        ax.set_title('Day ' + str(n+1), fontsize=20)
        ax.yaxis.set_ticks(ticks)
        ax.grid()
        ax.annotate('max = '+str(max_plot_separation), xy=(vmax*0.8, y_break*0.3), fontsize=20)
        ax.annotate('min = '+str(min_plot_separation), xy=(vmax*0.8, y_break*0.2), fontsize=20)

    e1u_masked = np.ma.array(e1u, mask=1 - umask)
    cmap = cm.Spectral_r
    cmap.set_bad('silver')
    p = ax0.pcolormesh(e1u_masked, cmap=cmap)
    cbar = fig.colorbar(p, ax=ax0, fraction=0.045, orientation='vertical', pad=0.009)
    cbar.ax.tick_params(labelsize=13)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax0.set_xlim([0, umask.shape[-1]])
    ax0.set_ylim([0, umask.shape[-2]])
    ax0.set_aspect(aspect='equal')
    ax0.set_title('Child Domain and e1u', fontsize=20)
    ax0.yaxis.set_ticks(ticks)
    ax0.yaxis.grid(color='k')

    fig.tight_layout(w_pad=1.2, h_pad=0.5)
    fig.suptitle(case+' - Separation '+kind_index+' for every y index at depth = '+str(int(depthu[dep_ind]))+' m', fontsize=25)
    plt.subplots_adjust(top=0.95)
    
    return fig

# --------------------------------------------------------------------------------------

def get_speeds_nstg(vozocrtx, vomecrty, umask, vmask, arrow, days):
    x_slice, y_slice, u_nstg, v_nstg, speeds = quicklook.get_speeds(vozocrtx, vomecrty, arrow)
    speeds_daily = quicklook.get_1day_avg_of_speeds(speeds, days)
    u_nstg_daily = quicklook.get_1day_avg_of_speeds(u_nstg, days)
    v_nstg_daily = quicklook.get_1day_avg_of_speeds(v_nstg, days)
    
    if arrow==1:
        umask_slice = umask[:-1, :-1]
        vmask_slice = vmask[:-1, :-1]
    else:
        umask_slice = umask[::arrow, ::arrow]
        vmask_slice = vmask[::arrow, ::arrow]
    
    return x_slice, y_slice, speeds_daily, u_nstg_daily, v_nstg_daily, umask_slice, vmask_slice

#

def plot_speed_combination(vozocrtx, vomecrty, umask, vmask, depthu, dep_ind, case):
    days = 9
    levels = [0.03, 0.05, 0.1, 0.2, 0.3]
    
    colour_list = ["#c8274c","#f25546","#F06543","#e96e33","#f0b038","#FFE74C",
                   "#69b944","#72b286","#69b0bc","#619ee4","#4b5bbb"][::-1]
    
    x_slice, y_slice, speeds_daily, u_nstg_daily, v_nstg_daily,\
        umask_slice, vmask_slice = get_speeds_nstg(vozocrtx, vomecrty, umask, 1, days)
    x_sliceq, y_sliceq, speeds_dailyq, u_nstg_dailyq, v_nstg_dailyq,\
        umask_sliceq, vmask_sliceq = get_speeds_nstg(vozocrtx, vomecrty, umask, 10, days)
    
    cmap = LinearSegmentedColormap.from_list('mycmap', colour_list, N=500, gamma=1)
    cmap.set_bad('silver')
    vmin, vmax = 0, speeds_daily.max()

    fig, axes = plt.subplots(3, 3, figsize = (20, 21))
    for ax, n in zip(axes.flatten(), np.arange(9)):
        
        plot_speeds = np.ma.array(speeds_daily[n, ...], mask=1 - umask_slice)
        plot_u_nstg = np.ma.array(u_nstg_daily[n, ...], mask=1 - umask_slice)
        plot_v_nstg = np.ma.array(v_nstg_daily[n, ...], mask=1 - vmask_slice)
        
        plot_speedsq = np.ma.array(speeds_dailyq[n, ...], mask=1 - umask_sliceq)
        plot_u_nstgq = np.ma.array(u_nstg_dailyq[n, ...], mask=1 - umask_sliceq)
        plot_v_nstgq = np.ma.array(v_nstg_dailyq[n, ...], mask=1 - vmask_sliceq)
        
        #
        p = ax.pcolormesh(x_slice, y_slice, plot_speeds, cmap=cmap, vmin=vmin, vmax=vmax)
        cs = ax.contour(x_slice, y_slice, plot_speeds, levels = levels, colors='k', alpha=0.5)
        #
        ax.quiver(x_sliceq, y_sliceq, plot_u_nstgq, plot_v_nstgq, color='k', 
                  clim=[vmin,vmax], pivot='mid', width=0.005, headwidth=2.5)
        #
        ax.clabel(cs, inline=1, fontsize=10)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_aspect(aspect='equal')
        ax.set_xlim([0, umask.shape[-1]])
        ax.set_ylim([0, umask.shape[-2]])
        ax.set_title('Day ' + str(n+1), fontsize=20)
        cbar = fig.colorbar(p, ax=ax, fraction=0.05, orientation='horizontal', pad=0.009)
        cbar.ax.tick_params(labelsize=13)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

    fig.tight_layout(w_pad=1.2, h_pad=0.01)
    fig.suptitle(case+' - Speeds [m/s] at depth = '+str(int(depthu[dep_ind]))+' m', fontsize=25)
    plt.subplots_adjust(top=0.96)
    return fig


# --------------------------------------------------------------------------------------