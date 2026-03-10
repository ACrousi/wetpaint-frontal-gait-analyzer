import numpy as np


def multi_input(data, conn):
    C, T, V, M = data.shape
    data_new = np.zeros((4, C*2, T, V, M))

    # Stream 0: Joint (absolute coords + relative to center joint)
    data_new[0,:C,:,:,:] = data
    for i in range(V):
        data_new[0,C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]

    # Stream 1: Velocity (1st-order motion + 2-frame displacement)
    for i in range(T-2):
        data_new[1,:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        data_new[1,C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]

    # Stream 2: Bone (bone vectors + bone angles)
    for i in range(len(conn)):
        data_new[2,:C,:,i,:] = data[:,:,i,:] - data[:,:,conn[i],:]
    bone_length = 0
    for i in range(C):
        bone_length += np.power(data_new[2,i,:,:,:], 2)
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        data_new[2,C+i,:,:,:] = np.arccos(data_new[2,i,:,:,:] / bone_length)

    # Stream 3: Acceleration (2nd-order finite difference + normalized direction)
    # a_t = x_{t+2} - 2*x_{t+1} + x_t
    for i in range(T-2):
        data_new[3,:C,i,:,:] = data[:,i+2,:,:] - 2*data[:,i+1,:,:] + data[:,i,:,:]
    acc_magnitude = 0
    for i in range(C):
        acc_magnitude += np.power(data_new[3,i,:,:,:], 2)
    acc_magnitude = np.sqrt(acc_magnitude) + 0.0001
    for i in range(C):
        data_new[3,C+i,:,:,:] = data_new[3,i,:,:,:] / acc_magnitude

    return data_new
