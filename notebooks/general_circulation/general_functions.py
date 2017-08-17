import matplotlib.pyplot as plt
import os,fnmatch
import numpy as np

# ------------------------------------------------------------------------------------------------

def get_files(dirname, fname, grid):
    ''' Sorts all output files.
    '''
    files = []
    for item in os.listdir(dirname):
        if fnmatch.fnmatchcase(item, fname + grid + "*.nc"):
            files += [os.path.join(dirname, item)]
    files.sort(key=os.path.basename)
    return files

# ------------------------------------------------------------------------------------------------

def set_plots(fig, axes, axa):
    ''' Applies formatting to all subplots.
    Applies unique formatting to the wind subplot,
    including labels. Applies spacings in figure.
    '''
    for ax in axes:
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor('gray')
    
    for pos in ['top', 'bottom', 'right', 'left']:
        axa.spines[pos].set_edgecolor('gray')
        axa.spines[pos].set_visible(False) 
        
    axa.set_xlabel('Time [hrs]', fontsize=13)
    axa.set_ylabel('Wind Stress [Nm$^{-2}$]', fontsize=13)
    axa.grid(b=True, which='major', color='gray', linestyle='-')
    axa.xaxis.grid(b=True, which='minor', color='gray', linestyle='-')
    axa.minorticks_on()
    axa.xaxis.set_ticks_position('none')
    axa.yaxis.set_ticks_position('none')
    
    plt.tight_layout(h_pad=0.9, w_pad=0.9, rect=[0, 0, 1, 0.96])
    return fig, axes, axa

# ------------------------------------------------------------------------------------

def set_plots_after_clear(ax, title, depm, fxn):
    ''' Re-applies formatting and labels after the 
    subplots are cleared so that the animation does not
    draw over a previous snapshot.
    '''
    ax.set_xlabel('X Indices (Child)', fontsize=13)
    if fxn == 'top':
        ax.set_ylabel('Y Indices (Child)', fontsize=13, labelpad=0.1)
    elif fxn == 'cross':
        ax.set_ylabel('Z Indices (Child)', fontsize=13, labelpad=0.1)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_title(title + str(int(depm)), fontsize=20)
    return ax

# ------------------------------------------------------------------------------------

def set_xy(ax, var_array):
    ''' Makes x and y arrays and applies limits
    to top view subplots.
    '''
    xs = np.arange(var_array.shape[-1])
    ys = np.arange(var_array.shape[-2])
    ax.set_xlim([0, xs[-1]]); ax.set_ylim([0, ys[-1]])
    return ax, xs, ys

# ------------------------------------------------------------------------------------

def set_yz(ax, var_array):
    ''' Makes y and z arrays and applies limits
    to cross section subplots.
    '''
    ys = np.arange(var_array.shape[-1])
    zs = np.arange(var_array.shape[-2])
    ax.set_xlim([0, ys[-1]]); ax.set_ylim([zs[-1],0])
    return ax, ys, zs

# ------------------------------------------------------------------------------------

def get_limits(var_arrayA, var_arrayB, var_arrayC, scale, lines, flag):
    ''' Finds the minimum and maximum values amongst three different arrays.
    For velocity or anomaly plots, it is preferrable to use a set of values for
    the colormap and contour levels that diverge from zero. As such, we use the
    absolute biggest value as the edge for this range. However, for temperature 
    and salinity plots, it is best to use a range of values that range from the
    minimum to maximum, regardless of their sign.
    '''
    vm_min = max(var_arrayA.min(), var_arrayB.min(), var_arrayC.min(), key=abs)
    vm_max = max(var_arrayA.max(), var_arrayB.max(), var_arrayC.max(), key=abs)
    if flag == 'vel' or flag == 'anom':
        vm0 = max([abs(vm_max), abs(vm_min)])
        vm = scale * vm0
        levels0 = np.linspace(-1 * vm, vm, lines)
        levels = np.round(levels0,2).tolist()
    elif flag == 'temp' or flag == 'salt': 
        levels0 = np.linspace(vm_min, vm_max, lines)
        levels = np.round(levels0,2).tolist()
    return vm_min, vm_max, levels

# ------------------------------------------------------------------------------------

