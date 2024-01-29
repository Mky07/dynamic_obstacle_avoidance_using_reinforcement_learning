import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def create_action_spaces(max_vel_x=1.5, max_vel_z=0.78, sample_size_x=10, sample_size_z=20, is_backwards = False):
    """
    create action_spaces
    [-max_vel_x/2 , max_vel_x] will divided into sample_size_x
    [-max_vel_z , max_vel_z] will divided into sample_size_z 
    """
    
    min_vel_x = 0.0 
    if is_backwards:
        min_vel_x = -max_vel_x/2
    samples_x = np.linspace(min_vel_x, max_vel_x, sample_size_x)
    samples_x = np.round(samples_x, 3)
    samples_z = np.linspace(-max_vel_z, max_vel_z, sample_size_z)
    samples_z = np.round(samples_z, 3)

    if np.where(samples_x == 0)[0].size == 0:
        samples_x = np.append(samples_x, 0.0)

    if np.where(samples_z == 0)[0].size == 0:
        samples_z = np.append(samples_z, 0.0)

    action_spaces = {}
    idx = 0
    for i,x in enumerate(samples_x):
        #if (x>=0.1):
        action_spaces[idx] = (x,0.0)
        idx+=1
        for j,z in enumerate(samples_z):
            #if (x>=0.1 and abs(z)>=0.1) or (x==0 and z==0):
            action_spaces[idx] = (x,z)
            idx+=1
    
    #delete (0,0)
    idx = 0
    a_spaces = {}
    for key, value in action_spaces.items():
        if value != (0.0, 0.0):
            a_spaces[idx] = value
            idx+=1
            
    print("action spaces have created. {}".format(a_spaces))
    return a_spaces

def min_max_normalize(self, val, min_val, max_val):
    normalized_data = (val - min_val) / (max_val - min_val)
    return normalized_data
