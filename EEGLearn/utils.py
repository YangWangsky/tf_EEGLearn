#coding:utf-8

import numpy as np
import math as m
import os
import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from functools import reduce


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r     tant^(-1)(y/x)
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian 
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates    [x, y, z]
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def load_data(data_file, classification=True):
    """                                               
    Loads the data from MAT file. MAT file should contain two
    variables. 'featMat' which contains the feature matrix in the
    shape of [samples, features] and 'labels' which contains the output
    labels as a vector. Label numbers are assumed to start from 1.

    Parameters
    ----------
    data_file: str
                        # load data from .mat [samples, (features:labels)]
    Returns 
    -------
    data: array_like
    """
    print("Loading data from %s" % (data_file))
    dataMat = scipy.io.loadmat(data_file, mat_dtype=True)
    print("Data loading complete. Shape is %r" % (dataMat['features'].shape,))
    if classification:
        return dataMat['features'][:, :-1], dataMat['features'][:, -1] - 1
    else:
        return dataMat['features'][:, :-1], dataMat['features'][:, -1]


def reformatInput(data, labels, indices):
    """
    Receives the indices for train and test datasets.
    param indices: tuple of (train, test) index numbers
    Outputs the train, validation, and test data and label datasets.
    """
    np.random.shuffle(indices[0])
    np.random.shuffle(indices[0])
    trainIndices = indices[0][len(indices[1]):]
    validIndices = indices[0][:len(indices[1])]
    testIndices = indices[1]

    if data.ndim == 4:
        return [(data[trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]
    elif data.ndim == 5:
        return [(data[:, trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[:, validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[:, testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    if inputs.ndim == 4:
        input_len = inputs.shape[0]
    elif inputs.ndim == 5:
        input_len = inputs.shape[1]
    assert input_len == len(targets)

    if shuffle:
        indices = np.arange(input_len)  
        np.random.shuffle(indices) 
    for start_idx in range(0, input_len, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if inputs.ndim == 4:
            yield inputs[excerpt], targets[excerpt]
        elif inputs.ndim == 5:
            yield inputs[:, excerpt], targets[excerpt]


def gen_images(locs, features, n_gridpoints=32, normalize=True, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] // nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])  # features.shape为[samples, 3*nElectrodes]

    nSamples = features.shape[0]    # sample number 2670
    # Interpolate the values        # print(np.mgrid[-1:1:5j]) get [-1.  -0.5  0.   0.5  1. ]
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))

    
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    
    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),    # cubic
                                    method='cubic', fill_value=np.nan)
    
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        
        temp_interp[c] = np.nan_to_num(temp_interp[c])
        
    temp_interp = np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H] # WH xy
    temp_interp = np.swapaxes(temp_interp, 1, 2)
    temp_interp = np.swapaxes(temp_interp, 2, 3)    # [samples, W, H，colors]
    return temp_interp



def load_or_generate_images(file_path, average_image=3):
    """
    Generates EEG images
    :param average_image: average_image 1 for CNN model only, 2 for multi-frame model 
                        sucn as lstm, 3 for both.

    :return:            Tensor of size [window_size, samples, W, H, channel] containing generated
                        images.
    """
    print('-'*100)
    print('Loading original data...')
    locs = scipy.io.loadmat('../SampleData/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    # Class labels should start from 0
    feats, labels = load_data('../SampleData/FeatureMat_timeWin.mat')   # 2670*1344 和 2670*1
    

    if average_image == 1:   # for CNN only
        if os.path.exists(file_path + 'images_average.mat'):
            images_average = scipy.io.loadmat(file_path + 'images_average.mat')['images_average']
            print('\n')
            print('Load images_average done!')
        else:
            print('\n')
            print('Generating average images over time windows...')
            # Find the average response over time windows
            for i in range(7):
                if i == 0:
                    temp  = feats[:, i*192:(i+1)*192]    # each window contains 64*3=192 data
                else:
                    temp += feats[:, i*192:(i+1)*192]
            av_feats = temp / 7
            images_average = gen_images(np.array(locs_2d), av_feats, 32, normalize=False)
            scipy.io.savemat( file_path+'images_average.mat', {'images_average':images_average})
            print('Saving images_average done!')
        
        del feats
        images_average = images_average[np.newaxis,:]
        print('The shape of images_average.shape', images_average.shape)
        return images_average, labels
    
    elif average_image == 2:    # for mulit-frame model such as LSTM
        if os.path.exists(file_path + 'images_timewin.mat'):
            images_timewin = scipy.io.loadmat(file_path + 'images_timewin.mat')['images_timewin']
            print('\n')    
            print('Load images_timewin done!')
        else:
            print('Generating images for all time windows...')
            images_timewin = np.array([
                gen_images(
                    np.array(locs_2d),
                    feats[:, i*192:(i+1)*192], 32, normalize=False) for i in range(feats.shape[1]//192)
                ])
            scipy.io.savemat(file_path + 'images_timewin.mat', {'images_timewin':images_timewin})
            print('Saving images for all time windows done!')
        
        del feats
        print('The shape of images_timewin is', images_timewin.shape)   # (7, 2670, 32, 32, 3)
        return images_timewin, labels
    
    else:
        if os.path.exists(file_path + 'images_average.mat'):
            images_average = scipy.io.loadmat(file_path + 'images_average.mat')['images_average']
            print('\n')
            print('Load images_average done!')
        else:
            print('\n')
            print('Generating average images over time windows...')
            # Find the average response over time windows
            for i in range(7):
                if i == 0:
                    temp = feats[:, i*192:(i+1)*192]
                else:
                    temp += feats[:, i*192:(i+1)*192]
            av_feats = temp / 7
            images_average = gen_images(np.array(locs_2d), av_feats, 32, normalize=False)
            scipy.io.savemat( file_path+'images_average.mat', {'images_average':images_average})
            print('Saving images_average done!')

        if os.path.exists(file_path + 'images_timewin.mat'):
            images_timewin = scipy.io.loadmat(file_path + 'images_timewin.mat')['images_timewin']
            print('\n')    
            print('Load images_timewin done!')
        else:
            print('\n')
            print('Generating images for all time windows...')
            images_timewin = np.array([
                gen_images(
                    np.array(locs_2d),
                    feats[:, i*192:(i+1)*192], 32, normalize=False) for i in range(feats.shape[1]//192)
                ])
            scipy.io.savemat(file_path + 'images_timewin.mat', {'images_timewin':images_timewin})
            print('Saving images for all time windows done!')

        del feats
        images_average = images_average[np.newaxis,:]
        print('The shape of labels.shape', labels.shape)
        print('The shape of images_average.shape', images_average.shape)    # (1, 2670, 32, 32, 3)
        print('The shape of images_timewin is', images_timewin.shape)   # (7, 2670, 32, 32, 3)
        return images_average, images_timewin, labels