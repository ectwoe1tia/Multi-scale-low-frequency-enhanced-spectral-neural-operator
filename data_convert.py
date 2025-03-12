import numpy as np
import h5py
import scipy

def mergedata_NavierStokes(filename, s):

    # Convert NavierStokes data to hdf5 file
    # s is resolution which equal to 64

    mat_data1 = scipy.io.loadmat(filename+'/NavierStokes_V1e-5_N1200_T20.mat')
    ntrain = 1000
    ntest = 200
    np.save(filename+'/train_v.npy', mat_data1['u'][:ntrain,:])
    np.save(filename+'/test_v.npy', mat_data1['u'][ntrain:,:])
    xcrd = np.linspace(0,1,s) 
    ycrd = np.linspace(0,1,s)
    nu = np.load(filename+'/test_v.npy')
    v = np.load(filename+'/train_v.npy')
    v = np.concatenate((nu,v),axis=0)
    np.save(filename+'/2D.npy', v)
    _beta = '_1.0'
    resolution = s
    savedir = filename
    flnm = savedir+'/2D_NS_' + '64' + _beta + '_Train.hdf5'
    with h5py.File(flnm, 'w') as f:
        f.create_dataset('tensor', data=np.load(savedir+'/2D.npy')[:, :, :, :, None])
        f.create_dataset('x-coordinate', data=xcrd)
        f.create_dataset('y-coordinate', data=ycrd)
    print(v.shape)

def mergedata_airfoils(filename, s1, s2):

    # Convert Pipe and Airfoils data to hdf5 files
    # When we use this program to convert airfoils tasks
    # s1 = 221 s2 = 51
    # When we use this program to convert pipe tasks, the following parameters need to be changed accordingly
    # ntest = 200, s1 = 129, s2 = 129
    # The path should be changed accordingly to Pipe_X.npy/Pipe_Y.npy/Pipe_Q.npy

    ntrain = 1000
    ntest = 100
    INPUT_X = np.load(filename + '/NACA_Cylinder_X.npy')[:, :, :, None]
    INPUT_Y = np.load(filename + '/NACA_Cylinder_Y.npy')[:, :, :, None]
    OUTPUT_Sigma = np.load(filename + '/NACA_Cylinder_Q.npy')
    output = OUTPUT_Sigma[:,4]
    input_X = np.stack([INPUT_X, INPUT_Y], axis=-1)

    train_x = input_X[ :ntrain, :, :]
    test_x = input_X[ ntrain:ntrain+ntest, :, :]
    Ma = np.concatenate((test_x,train_x),axis=0)
    
    train_y = output[ :ntrain, :, :]
    test_y = output[ ntrain:ntrain+ntest, :, :]
    tensor = np.concatenate((test_y,train_y),axis=0)

    xcrd = np.linspace(0,1,s1) 
    ycrd = np.linspace(0,1,s2)
    savedir = filename
    flnm = savedir+'/2D_NS_' + 'airfoils' + '_Train.hdf5'

    with h5py.File(flnm, 'w') as f:
        f.create_dataset('tensor', data=tensor)
        f.create_dataset('Ma', data=Ma)
        f.create_dataset('x-coordinate', data=xcrd)
        f.create_dataset('y-coordinate', data=ycrd)

    print(tensor.shape)
    print(Ma.shape)

def mergedata_elasticity_interp(filename, s):

    # Convert elasticity_G data to hdf5 file
    # s is resolution which equal to 41

    ntrain = 1000
    ntest = 200
    INPUT = np.load(filename + './Interp/Random_UnitCell_mask_10_interp.npy')
    OUTPUT = np.load(filename + './Interp/Random_UnitCell_sigma_10_interp.npy')[:, :, :, None]
    INPUT = np.transpose(INPUT,(2,0,1))
    OUTPUT = np.transpose(OUTPUT,(2,3,0,1))
    
    train_x = INPUT[ :ntrain, :, :]
    test_x = INPUT[ -ntest:, :, :]
    nu = np.concatenate((test_x,train_x),axis=0)
    
    train_y = OUTPUT[ :ntrain, :, :]
    test_y = OUTPUT[ -ntest:, :, :]
    tensor = np.concatenate((test_y,train_y),axis=0)

    xcrd = np.linspace(0,1,s) 
    ycrd = np.linspace(0,1,s)
    savedir = filename
    flnm = savedir+'/2D_Elasticity_' + 'interp' + '_Train.hdf5'

    with h5py.File(flnm, 'w') as f:
        f.create_dataset('tensor', data=tensor)
        f.create_dataset('nu', data=nu)
        f.create_dataset('x-coordinate', data=xcrd)
        f.create_dataset('y-coordinate', data=ycrd)

    print(tensor.shape)
    print(nu.shape)