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

# --------------------------------------------------------------------------------------

def plot_speed_pcolormesh(x_slice, y_slice, speeds_daily, umask, levels, depthu, dep_ind, arrow):
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
    fig.suptitle('Speeds [m/s] at depth = '+str(int(depthu[dep_ind]))+' m', fontsize=25)
    plt.subplots_adjust(top=0.96)
    return fig

# --------------------------------------------------------------------------------------

def plot_speed_quiver(x_slice, y_slice, u_nstg_daily, v_nstg_daily, speeds_daily, umask, vmask, levels, depthu, dep_ind, arrow):
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
    fig.suptitle('Speeds [m/s] at depth = '+str(int(depthu[dep_ind]))+' m', fontsize=25)
    plt.subplots_adjust(top=0.96)
    return fig

# --------------------------------------------------------------------------------------