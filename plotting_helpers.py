import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import gaussian_filter

border_ratio = 0.12

global arena_d

title_font = {
        'family': 'sans',    
        'size': 23,
        'horizontalalignment': 'left'
        }


panel_label_font = {
        'family': 'sans',    
        'size': 18,
        'horizontalalignment': 'center'
        }

hex_colors = [
    "#4363d8",
    "#3cb44b",
    "#e6194B",
    "#ffe119",
    "#f032e6",
    "#f58231",
    "#42d4f4",
    "#fabebe",
    "#e6beff",
    "#469990",
    "#a9a9a9",
    "#800000",
]
rgb_colors = [    
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 225, 25),
    (240, 50, 230),
    (245, 130, 49),
    (66, 212, 244),
    (250, 190, 190),
    (230, 190, 255),
    (70, 153, 144),
    (169, 169, 169),
    (128, 0, 0),
    (161, 5, 13),
    (8, 192, 51),
    (182, 73, 4),
    (174, 70, 108),
    (171, 87, 25),
    (212, 85, 24),
    (228, 156, 151),
    (164, 91, 18),
    (25, 214, 23),
    (168, 123, 216),
    (225, 200, 201),
    (187, 108, 175),
    (197, 156, 213),
    (106, 152, 86),
    (24, 51, 196),
    (42, 34, 252),
    (220, 99, 81),
    (180, 50, 154),
    (188, 95, 114),
    (33, 91, 158),
    (148, 181, 5),
    (16, 99, 20),
    (26, 44, 111),
    (23, 161, 43),
    (162, 151, 79),
    (201, 131, 225),
    (46, 122, 241),
    (179, 22, 117),
    (106, 73, 12),
    (77, 168, 188),
    (122, 253, 222),
    (112, 156, 85),
    (173, 240, 184),
    (252, 6, 95),
    (158, 71, 120),
    (109, 154, 248)
]

colormaps = {
    'autumn' : cv2.COLORMAP_HOT,
    'bone' : cv2.COLORMAP_BONE,
    'cool' : cv2.COLORMAP_COOL,
    'hot' : cv2.COLORMAP_HOT,
    'hsv' : cv2.COLORMAP_HSV,
    'jet' : cv2.COLORMAP_JET,
    'ocean' : cv2.COLORMAP_OCEAN, 
    'pink' : cv2.COLORMAP_PINK,
    'raibow' : cv2.COLORMAP_RAINBOW,
    'spring' : cv2.COLORMAP_SPRING,
    'summer' : cv2.COLORMAP_SUMMER,
    'winter' : cv2.COLORMAP_WINTER
}
def toggle_spines(value):
    plt.rcParams['axes.spines.left'] = value
    plt.rcParams['axes.spines.right'] = value
    plt.rcParams['axes.spines.top'] = value
    plt.rcParams['axes.spines.bottom'] = value


def get_layout(n):
    n_sqrtf = np.sqrt(n)
    n_sqrt = int(np.ceil(n_sqrtf))
    if n_sqrtf == n_sqrt:
        x, y = n_sqrt, n_sqrt
    elif n <= n_sqrt * (n_sqrt - 1):
        x, y = n_sqrt, n_sqrt - 1
    elif not (n_sqrt % 2) and n % 2:
        x, y = (n_sqrt + 1, n_sqrt - 1)
    else:
        x, y = n_sqrt, n_sqrt
    if y == 1:
        return tuple([x, 1])
    if n == x * y:
        return tuple(x for i in range(y))
    if (x % 2) != (y % 2) and (x % 2):
        x, y = y, x    
    return (x, y)

def render_arena(exit_angle, plot_width, rotation_angle=0, room_centric = False, room_shading = True):
    global arena_d, border
    arena_d=int(plot_width/(1+2*border_ratio))    
    border=(plot_width - arena_d)//2
    centre_d=int(arena_d*0.12)
    if exit_angle is None:
        room_rotation = 0
        port_rotation = 0
        room_shading = False
    else:
        if room_centric:
            room_rotation = 0
            port_rotation = exit_angle
        else:
            room_rotation = -exit_angle
            port_rotation = 0
    if rotation_angle is not None:
        room_rotation += rotation_angle
        port_rotation += rotation_angle
    light_d = arena_d//2
    blur_d = arena_d//3
    bg_shade = 130
    arena_shade = 220
    centre_shade = 200
    light_shade = 200
    img = np.full((plot_width*2,plot_width*2,4),bg_shade, np.uint8)
    img[:,:,-1]=255
    if room_shading:
        cv2.circle(img, (plot_width//2*3,plot_width),light_d,(light_shade,light_shade,light_shade, 255), cv2.FILLED)
        img = cv2.blur(img, (blur_d, blur_d))
        rot_mat = cv2.getRotationMatrix2D((plot_width,plot_width), room_rotation, 1.0)
        img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    img = img[plot_width//2:plot_width//2*3, plot_width//2:plot_width//2*3, :]
    cv2.circle(img, (img.shape[0]//2,img.shape[0]//2),arena_d//2,(arena_shade,arena_shade,arena_shade, 255), cv2.FILLED)
    cv2.circle(img, (img.shape[0]//2,img.shape[0]//2),centre_d//2,(centre_shade,centre_shade,centre_shade, 255), cv2.FILLED)
    for alpha in range(0,360,45):    
        img_port = np.full((plot_width,plot_width,4), 0, np.uint8)
        port_coords = (np.array([0.475, 0.075, 0.525, 0.15, 0.5, 0.025]) * arena_d).astype(int)
        port_coords[0:5] += border
        cv2.rectangle(img_port,(port_coords[0], port_coords[1]),(port_coords[2], port_coords[3]),(255,255,255,255),cv2.FILLED)
        cv2.circle(img_port,(port_coords[4], port_coords[3]),port_coords[5],(255,255,255,255),cv2.FILLED)    
        if (alpha != port_rotation+180) & (alpha != port_rotation-180):
            cv2.ellipse(img_port,(port_coords[4]-1, port_coords[1]),(port_coords[5]-1,port_coords[5]-1),0,0,180,(0,0,0,255),cv2.FILLED)
        else:
            cv2.ellipse(img_port,(port_coords[4]-1, port_coords[1]),(port_coords[5]-1,port_coords[5]-1),0,0,180,(235,235,235,255),cv2.FILLED)
        rot_mat = cv2.getRotationMatrix2D((plot_width//2,plot_width//2), alpha, 1.0)
        img_port = cv2.warpAffine(img_port, rot_mat, img_port.shape[1::-1], flags=cv2.INTER_LINEAR)
        img[img_port[...,3] == 255] = img_port[img_port[...,3] == 255]        
    return img[:,:,:3].copy()    


def draw_mouse(img, coords, filled, color_rgb):
    centre = (int((coords['X_centre'] + 1) * arena_d / 2 + border), int((coords['Y_centre'] + 1) * arena_d / 2 + border))
    nose = (int((coords['X_nose']) * arena_d / 2), int((coords['Y_nose']) * arena_d / 2))
    tail = (int((coords['X_tail']) * arena_d / 2), int((coords['Y_tail']) * arena_d / 2))
    radius = int(np.sqrt(abs(coords['Area']) / np.pi) * 8)
    taillength = np.sqrt(np.sum(np.power(tail, 2))) + 0.1
    tail_norm = np.divide(tail, taillength)
    tailtip = tuple(np.add(np.multiply(tail_norm, 32 + taillength * 0.5).astype(int), centre))
    cv2.circle(img, centre, radius, color_rgb, filled)
    cv2.circle(img, tuple(np.add(nose, centre)), 7, color_rgb, filled)
    cv2.circle(img, tuple(np.add(tail, centre)), 10, color_rgb, filled)
    cv2.line(img, tuple(np.add(tail, centre)), tailtip, color_rgb, thickness=2)
    return img


def draw_trajectory(img, coordses, time_steps, index, plot_type, color_var=None, landmark="centre"):
    centres = [(int((coords['X_centre'] + 1) * arena_d / 2 + border), int((coords['Y_centre'] + 1) * arena_d / 2 + border)) for index, coords in coordses.iterrows()]
    noses = [(int((coords['X_nose']) * arena_d / 2), int((coords['Y_nose']) * arena_d / 2)) for index, coords in coordses.iterrows()]
    tails = [(int((coords['X_tail']) * arena_d / 2), int((coords['Y_tail']) * arena_d / 2)) for index, coords in coordses.iterrows()]
    refpoints = [centres, centres + noses, centres + tails][['centre', 'nose', 'tail'].index(landmark)]
    time_color = (int(index / time_steps * 255), 0, 255 - int(index / time_steps * 255))
    if  plot_type=='path_colorvar':
        color_var_color = (int(color_var[index] * 255), 0, 255 - int(color_var[index] * 255))
    cv2.line(img, refpoints[0], refpoints[1], time_color if plot_type=='path_timecolor' else color_var_color)   
    return img

def draw_trajectories(img, trajectory, plot_width, plot_type='path_timecolor', color_id=0, color_var=None, landmark='centre', skip_mice=5, weights=None, colormap='hot', binsize=8):      
#     arena_d=int(plot_width/(1+2*border_ratio))
    if plot_type=='heatmap':
        xs = (trajectory['X_centre'] + trajectory['X_nose'] + 1) * arena_d / 2 + border
        ys = (trajectory['Y_centre'] + trajectory['Y_nose'] + 1) * arena_d / 2 + border
        binsize = binsize
        heatmap = np.histogram2d(ys, xs, range=[[0, plot_width],[0, plot_width]], bins=plot_width//binsize, weights=weights)[0].repeat(binsize,1).repeat(binsize,0)
        heatmap = heatmap / np.max(heatmap) * 255 * 4
        heatmap = np.array(heatmap, dtype = np.uint8)
        im_color = cv2.applyColorMap(heatmap.astype(np.uint8), colormaps[colormap])
        im_color = cv2.GaussianBlur(im_color, (0,0),8)
        cv2.addWeighted(img, 0.5, im_color, 0.9, 0, img)
        return img.copy()
    if plot_type=='path_timecolor':
        time_steps = trajectory.shape[0] - 1
        for index in range(1, time_steps):
            draw_trajectory(img, trajectory.iloc[index:index+2,:], time_steps, index, plot_type, landmark=landmark)
    if plot_type=='path_colorvar':
        time_steps = trajectory.shape[0] - 1
        for index in range(1, time_steps):
            draw_trajectory(img, trajectory.iloc[index:index+2,:], time_steps, index, plot_type, color_var, landmark)
    if plot_type=='mice':
        time_steps = trajectory.shape[0] - 1
        for index in range(1, time_steps, skip_mice):
            img = draw_mouse(img, trajectory.iloc[index,:], -1,  rgb_colors[color_id])        
    if plot_type=='mice_colorids':
        time_steps = trajectory.shape[0] - 1
        for index in range(1, time_steps, skip_mice):
            img = draw_mouse(img, trajectory.iloc[index,:], -1,  rgb_colors[color_id[int(index)]])
    if plot_type=='mice_colorvar':
        time_steps = trajectory.shape[0] - 1
        for index in range(1, time_steps, skip_mice):
            img = draw_mouse(img, trajectory.iloc[index,:], -1, (int(color_var[int(index)] / max(color_var) * 255), 0, 255 - int(color_var[index] / max(color_var) * 255)))
    if plot_type=='mice_timecolor':
        time_steps = trajectory.shape[0] - 1
        for index in range(1, time_steps, skip_mice):
            img = draw_mouse(img, trajectory.iloc[index,:], -1, (int(index / time_steps * 255), 0, 255 - int(index / time_steps * 255)))    
    return img.copy()   
