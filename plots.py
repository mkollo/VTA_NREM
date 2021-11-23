from plotting_helpers import *

from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import os
import imageio
import subprocess
from ipywidgets import Image
from IPython import display
import math

def cluster_corr(clustering1, clustering2):
    cluster_corr = np.zeros((len(clustering1),len(clustering2)))
    for i, c1 in enumerate(clustering1):
        for j, c2 in enumerate(clustering2):
            cluster_corr[i, j]=(sum([c in c1 for c in c2])+sum([c in c2 for c in c1]))/(2*(len(c1)+len(c2)))
    return cluster_corr

def plot_cluster_corr(clustering1, clustering2, label1="", label2=""):
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    fig, ax = plt.subplots(1,1)
    cl_arr=cluster_corr(clustering1, clustering2)    
    ax.imshow(cl_arr, cmap="Reds", vmin=0, vmax=1, aspect=1, interpolation='nearest')
    ax.set_xlim([-0.5,max(clustering2)-0.5])
    ax.set_ylim([-0.5,max(clustering1)-0.5])
    ax.set_xticks(range(cl_arr.shape[1]))
    ax.set_yticks(range(cl_arr.shape[0]))
    ax.set_xticklabels(range(1,cl_arr.shape[1]+1))
    ax.set_yticklabels(range(1,cl_arr.shape[0]+1))
    ax.set_xlabel(label2)     
    ax.set_ylabel(label1)
    ax.invert_yaxis()
    plt.show()
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False


def reconstruction(input_data, output_data, plot_type='mice_timecolor', filename=None, col_name='Trial', batch_size=None):          
    sample_indices=list(range(0,input_data.shape[0],input_data.shape[0]//5))   
    m=len(sample_indices)    
    _,n,r = input_data.shape
    input_df = pd.DataFrame(np.column_stack((np.repeat(np.arange(m),n),input_data[sample_indices,:,:].reshape(m*n,-1))), columns=[col_name, 'X_centre', 'Y_centre', 'X_nose', 'Y_nose', 'X_tail', 'Y_tail', 'Area'])
    input_df.insert(0, 'Layer', 'Input')
    output_df = pd.DataFrame(np.column_stack((np.repeat(np.arange(m),n),output_data[sample_indices,:,:].reshape(m*n,-1))), columns=[col_name, 'X_centre', 'Y_centre', 'X_nose', 'Y_nose', 'X_tail', 'Y_tail', 'Area'])
    output_df.insert(0, 'Layer', 'Output')
    all_df=input_df.append(output_df)
    all_df['Exit_angle']=0
    all_df[col_name]=all_df[col_name].astype(int)
    trajectory(all_df, group_by=['Layer', col_name], plot_type=plot_type, filename=filename)

def save_gif(images, filepath, speed):
    tempfile.tempdir = "/dev/shm"
    fd, path = tempfile.mkstemp()
    try:
        print("\nSaving temporary GIF...", flush=True)
        imageio.mimsave(
            path, images, format="GIF", duration=1/speed, palettesize=16, subrectangles=True
        )
        print("\nCompressing GIF...", flush=True)
        subprocess.run(
            [
                "gifsicle",
                "--verbose",
                "--no-conserve-memory",
                "-O3",
                "--lossy=90",
                "-o",
                filepath,
                path,
            ]
        )
        print("\nGIF compressed", flush=True)
    finally:
        pass
        os.remove(path)
    return True

def rotate(data, angle):
    complex_points_centre = data.iloc[:,4] + 1j * data.iloc[:,5]
    complex_result_centre = complex_points_centre * np.exp(complex(0, math.radians(angle)))
    complex_points_nose = data.iloc[:,6] + 1j * data.iloc[:,7]
    complex_result_nose = complex_points_nose * np.exp(complex(0, math.radians(angle)))
    complex_points_tail = data.iloc[:,8] + 1j * data.iloc[:,9]
    complex_result_tail = complex_points_tail * np.exp(complex(0, math.radians(angle)))
    rotated_data=np.stack((data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],data.iloc[:,3],np.real(complex_result_centre),np.imag(complex_result_centre),np.real(complex_result_nose),np.imag(complex_result_nose),np.real(complex_result_tail),np.imag(complex_result_tail), data.iloc[:,10],data.iloc[:,11],data.iloc[:,12]),-1)
    return pd.DataFrame(rotated_data, index=data.index, columns=data.columns)

def trajectory(data, plot_width=512, group_by=[], rotation_angle = None, max_groups=None, plot_type='path_timecolor', color_id=0, color_var=None, skip_mice=7, filename=None, fig_scale=15, ax=None, speed=100, weights=None, colormap='hot', binsize=8):
    if isinstance(data, np.ndarray):
        data_df = pd.DataFrame(columns=['Trial', '_', 'X_centre', 'Y_centre', 'X_nose', 'Y_nose', 'X_tail', 'Y_tail', 'Area'])
        for i in range(data.shape[0]):
            trial_df = pd.DataFrame(data[i,:,:], columns=['X_centre', 'Y_centre', 'X_nose', 'Y_nose', 'X_tail', 'Y_tail', 'Area'])
            trial_df['Trial']=i
            trial_df['_']=' '
            data_df=data_df.append(trial_df)
            group_by=['_', 'Trial']
        data=data_df
        data["Exit_angle"] = None
    if plot_type=="animation":
        exit_angle = data["Exit_angle"].iloc[0] 
        imgs = []
        time_steps = data.shape[0] - 2
        for time_step in range(1, time_steps):
            workdone = time_step / time_steps
            print(
                "\rGIF generation: [{0:50s}] {1:.1f}%".format("#" * int(workdone * 50), workdone * 100),
                end="",
                flush=True,
            )
            img = render_arena(exit_angle, plot_width, rotation_angle=rotation_angle)
            trajectory = data.iloc[time_step-1:time_step+2,:]
            if rotation_angle is not None:
                            trajectory = rotate(trajectory, rotation_angle)
            if color_var is not None:
                step_color_var = color_var[time_step-1:time_step+2]
                step_color_var.append(max(color_var))
                img = draw_trajectories(img, trajectory, plot_width, plot_type="mice_colorvar", skip_mice=1, color_var=step_color_var, colormap=colormap, binsize=binsize)
            else:    
                if type(color_id)==list:
                    img = draw_trajectories(img, trajectory, plot_width, plot_type="mice", skip_mice=1, color_id=color_id[time_step], color_var=color_var, colormap=colormap, binsize=binsize)
                else:
                    img = draw_trajectories(img, trajectory, plot_width, plot_type="mice", skip_mice=1, color_id=color_id, color_var=color_var, colormap=colormap, binsize=binsize)
            imgs.append(img.copy())
        save_gif(imgs, filename, speed=speed)         
#         file = open(filename , "rb")
#         image = file.read()
#         progress= Image(
#             value=image,
#             format='gif',
#             width=plot_width,
#             height=plot_width)
#         display.display(progress)
    else:
        toggle_spines(False)
        if len(group_by)==2:
            group_names = (data[group_by[0]].unique(), data[group_by[1]].unique())
    #         Paired groups
            if len(group_names[0]) * len(group_names[1]) == len([group for group in data.groupby(group_by)]):
                n_rows = len(group_names[0])
                n_cols = len(group_names[1])
                fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_scale, int(fig_scale/n_rows)), constrained_layout=True)
                if n_rows==1:
                    for c in range(n_cols):
                        if c==0:
                            category_label = ""
                            ax[c].set_ylabel(group_by[0]+" "+category_label, rotation=0, ha='right')
                        category_label = group_names[1][c]  if isinstance(group_names[1][c], str) else str(group_names[1][c] + 1)
                        ax[c].set_xlabel(group_by[1]+" "+category_label, rotation=0, ha='center')
                        ax[c].xaxis.set_label_position('top')
                        row_value = " "                    
                        col_value = group_names[1][c]
                        trajectory=data[(data[group_by[0]]==row_value) & (data[group_by[1]]==col_value)]
                        if rotation_angle is not None:
                            trajectory = rotate(trajectory, rotation_angle)
                        exit_angle = trajectory["Exit_angle"].iloc[0]
                        img = render_arena(exit_angle, plot_width, rotation_angle=rotation_angle)              
                        draw_trajectories(img, trajectory, plot_width, plot_type=plot_type, skip_mice=skip_mice, color_id=color_id, color_var=color_var, weights=weights, colormap=colormap, binsize=binsize)
                        ax[c].imshow(img)
                        ax[c].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)                  
                else:
                    for r in range(n_rows):                
                        for c in range(n_cols):
                            if c==0:
                                category_label = group_names[0][r]  if isinstance(group_names[0][r], str) else str(group_names[0][r] + 1)
                                ax[r, c].set_ylabel(group_by[0]+" "+category_label, rotation=0, ha='right')
                            if r==0:
                                category_label = group_names[1][c]  if isinstance(group_names[1][c], str) else str(group_names[1][c] + 1)
                                ax[r, c].set_xlabel(group_by[1]+" "+category_label, rotation=0, ha='center')
                            ax[r, c].xaxis.set_label_position('top')
                            row_value = group_names[0][r]                    
                            col_value = group_names[1][c]
                            trajectory=data[(data[group_by[0]]==row_value) & (data[group_by[1]]==col_value)]    
                            if rotation_angle is not None:
                                trajectory = rotate(trajectory, rotation_angle)
                            exit_angle = trajectory["Exit_angle"].iloc[0]
                            img = render_arena(exit_angle, plot_width, rotation_angle=rotation_angle)              
                            draw_trajectories(img, trajectory, plot_width, plot_type=plot_type, skip_mice=skip_mice, color_id=color_id, color_var=color_var, weights=weights, colormap=colormap, binsize=binsize)
                            ax[r, c].imshow(img)
                            ax[r, c].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)                     
    #         Unpaired groups
            else:            
                n_rows = len(group_names[0])
                n_cols = [data[data[group_by[0]]==g][group_by[1]].unique().shape[0] for g in group_names[0]]
                fig, ax = plt.subplots(n_rows, max(n_cols), figsize=(20, 10))
                for row_i, row_n_cols in enumerate(n_cols):
                    ax[row_i, 0].set_title(group_names[0][row_i], fontdict=title_font, loc='left', pad=10)
                    for c in range(row_n_cols):                                        
                        row_value = group_names[0][row_i]                    
                        col_value = data[data[group_by[0]]==row_value][group_by[1]].unique()[c]               
                        trajectory=data[(data[group_by[0]]==row_value) & (data[group_by[1]]==col_value)]
                        if rotation_angle is not None:
                            trajectory = rotate(trajectory, rotation_angle)
                        exit_angle = trajectory["Exit_angle"].iloc[0]  
                        img = render_arena(exit_angle, plot_width, rotation_angle=rotation_angle)
                        draw_trajectories(img, trajectory, plot_width, plot_type=plot_type, skip_mice=skip_mice, color_id=color_id, color_var=color_var, weights=weights, colormap=colormap, binsize=binsize)
                        ax[row_i, c].imshow(img)
                        ax[row_i, c].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
                        ax[row_i, c].set_xlabel(group_by[1] + " " + str(col_value + 1), fontdict=panel_label_font)                                
                    for c in range(row_n_cols, max(n_cols)):
                        ax[row_i, c].axis('off')     
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 5))
            exit_angle = data["Exit_angle"].iloc[0] 
            img = render_arena(exit_angle, plot_width, rotation_angle=rotation_angle)
            trajectory = data
            if rotation_angle is not None:
                trajectory = rotate(trajectory, rotation_angle)
            img = draw_trajectories(img, trajectory, plot_width, plot_type=plot_type, skip_mice=skip_mice, color_id=color_id, color_var=color_var, weights=weights, colormap=colormap, binsize=binsize)
    #         plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            imgplot = ax.imshow(img)
            ax.axis('off')
        if filename is None:    
            toggle_spines(True)
        else:
            plt.savefig(filename)    
