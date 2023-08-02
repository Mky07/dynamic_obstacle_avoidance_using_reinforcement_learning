import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def create_action_spaces(max_vel_x=1.5, max_vel_z=0.78, sample_size_x=10, sample_size_z=20):
    """
    create action_spaces
    [-max_vel_x/2 , max_vel_x] will divided into sample_size_x
    [-max_vel_z , max_vel_z] will divided into sample_size_z 
    """

    samples_x = np.linspace(-max_vel_x/2, max_vel_x, sample_size_x)
    samples_z = np.linspace(-max_vel_z, max_vel_z, sample_size_z)

    action_spaces = {}
    idx = 0
    # is_zero = False
    for i,x in enumerate(samples_x):
        for j,z in enumerate(samples_z):
            action_spaces[idx] = (x,z)
            idx+=1
            # if x == 0.0 and z == 0.0:
            #     is_zero = True

    # if not is_zero:
    #     action_spaces[idx] = (0.0, 0.0)
    print("action spaces have created. {}".format(action_spaces))
    return action_spaces