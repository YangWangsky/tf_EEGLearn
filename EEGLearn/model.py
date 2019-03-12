##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yang Wang
## School of Automation, Huazhong University of Science & Technology (HUST)
## wangyang_sky@hust.edu.cn
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#coding:utf-8

import tensorflow as tf

def my_conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='same', activation=None, name=None, reuse=None):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), bias_initializer=tf.constant_initializer(0.1), name=name, reuse=reuse)
                
def build_cnn(input_image=None, image_size=32, n_colors=3, activation_function=tf.nn.relu, reuse=None, name='VGG_NET_CNN'):
    # VGG_NET 32       # [samples, W, H, colors]
    with tf.variable_scope(name, reuse=reuse): 
        input_image = tf.reshape(input_image, shape=[-1, image_size, image_size, n_colors], name='Reshape_inputs')
        # layer_1   # 4个3*3*32
        
        h_conv1_1 = my_conv2d(input_image, filters=32, kernel_size=(3,3), activation=activation_function, name='conv1_1')
        h_conv1_2 = my_conv2d(h_conv1_1, filters=32, kernel_size=(3,3), activation=activation_function, name='conv1_2')
        h_conv1_3 = my_conv2d(h_conv1_2, filters=32, kernel_size=(3,3), activation=activation_function, name='conv1_3')
        h_conv1_4 = my_conv2d(h_conv1_3, filters=32, kernel_size=(3,3), activation=activation_function, name='conv1_4')
        h_pool1 = tf.layers.max_pooling2d(h_conv1_4, pool_size=(2,2), strides=(2,2), padding='same', name='max_pooling_1')    # shape is (None, 16, 16, 32)

        # layer_2
        h_conv2_1 = my_conv2d(h_pool1, filters=64, kernel_size=(3,3), activation=activation_function, name='conv2_1')
        h_conv2_2 = my_conv2d(h_conv2_1, filters=64, kernel_size=(3,3), activation=activation_function, name='conv2_2')
        h_pool2 = tf.layers.max_pooling2d(h_conv2_2, pool_size=(2,2), strides=(2,2), padding='same', name='max_pooling_2')    # shape is (None, 8, 8, 64)

        # layer_3
        h_conv3_1 = my_conv2d(h_pool2, filters=128, kernel_size=(3,3), activation=activation_function, name='conv3_1')
        h_pool3 = tf.layers.max_pooling2d(h_conv3_1, pool_size=(2,2), strides=(2,2), padding='same', name='max_pooling_3')    # shape is (None, 4, 4, 128)

    return h_pool3


def build_convpool_max(input_image, nb_classes, image_size=32, n_colors=3, 
        n_timewin=7, dropout_rate=0.5, name='CNN_Max', train=True, reuse=False):
    """
    Builds the complete network with maxpooling layer in time.

    :param input_image: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param image_size: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    with tf.name_scope(name):
        with tf.name_scope('Parallel_CNNs'):
            convnets = []
            # Build 7 parallel CNNs with shared weights
            for i in range(n_timewin):
                if i==0:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=reuse)
                else:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=True)
                convnets.append(convnet)    # list contains [None, 4, 4, 128]
            convnets = tf.stack(convnets)   # [n_timewin, nSamples, 4, 4, 128]
            convnets = tf.transpose(convnets, [1,0,2,3,4]) # [nSamples, n_timewin, 4, 4, 128]
        
        with tf.variable_scope('Max_pooling_over_flames'):
            # convpooling using Max pooling over frames
            convnets = tf.reshape(convnets, shape=[ -1, n_timewin, 4*4*128, 1])
            convpool = tf.nn.max_pool(convnets, # [nSamples, 1，4*4*128, 1]
                ksize=[1, n_timewin, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='convpool_max')
        

        convpool_flat = tf.reshape(convpool, [-1, 4*4*128])
        h_fc1_drop1 = tf.layers.dropout(convpool_flat, rate=dropout_rate, training=train, name='dropout_1')
        # input shape [batch, 4*4*128] output shape [batch, 512]
        h_fc1 = tf.layers.dense(h_fc1_drop1, 512, activation=tf.nn.relu, name='fc_relu_512')
        # dropout 
        h_fc1_drop2 = tf.layers.dropout(h_fc1, rate=dropout_rate, training=train, name='dropout_2')
        # inputshape [batch, 512] output shape [batch, nb_classes]    # the loss function contains the softmax activation
        prediction = tf.layers.dense(h_fc1_drop2, nb_classes, name='fc_softmax')
    
    return prediction

def build_convpool_conv1d(input_image, nb_classes, image_size=32, n_colors=3, 
        n_timewin=7, dropout_rate=0.5, name='CNN_Conv1d', train=True, reuse=False):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.

    :param input_image: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param image_size: size of the input image (assumes a square input)S
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    with tf.name_scope(name):
        with tf.name_scope('Parallel_CNNs'):
            convnets = []
            # Build 7 parallel CNNs with shared weights
            for i in range(n_timewin):
                if i==0:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=reuse)
                else:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=True)
                convnets.append(convnet)
            convnets = tf.stack(convnets)
            convnets = tf.transpose(convnets, [1,0,2,3,4])

        with tf.variable_scope('Conv1d_over_flames'):
            convnets = tf.reshape(convnets, shape=[ -1, n_timewin, 4*4*128, 1])
            convpool = my_conv2d(convnets, filters=64, kernel_size=(3, 4*4*128), strides=(1, 1), padding='valid', activation=tf.nn.relu, name='convpool_conv1d')


        with tf.variable_scope('Output_layers'):
            convpool_flat = tf.reshape(convpool, [-1, (n_timewin-2)*64])
            h_fc1_drop1 = tf.layers.dropout(convpool_flat, rate=dropout_rate, training=train, name='dropout_1')
            h_fc1 = tf.layers.dense(h_fc1_drop1, 256, activation=tf.nn.relu, name='fc_relu_256')
            h_fc1_drop2 = tf.layers.dropout(h_fc1, rate=dropout_rate, training=train, name='dropout_2')
            prediction = tf.layers.dense(h_fc1_drop2, nb_classes, name='fc_softmax')
    
    return prediction


def build_convpool_lstm(input_image, nb_classes, grad_clip=110, image_size=32, n_colors=3, 
        n_timewin=7, dropout_rate=0.5, num_units=128, batch_size=32, name='CNN_LSTM', train=True, reuse=False):
    """
    Builds the complete network with LSTM layer to integrate time from sequences of EEG images.

    :param input_image: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param image_size: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :param num_units: number of units in the LSTMCell
    :return: a pointer to the output of last layer
    """
    with tf.name_scope(name):
        with tf.name_scope('Parallel_CNNs'):
            convnets = []
            # Build 7 parallel CNNs with shared weights
            for i in range(n_timewin):
                if i==0:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=reuse)
                else:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=True)
                convnets.append(convnet)
            convnets = tf.stack(convnets)
            convnets = tf.transpose(convnets, [1,0,2,3,4]) # 调换轴 shape: (nSamples, n_timewin, 4, 4, 128)

        with tf.variable_scope('LSTM_layer'):
            # (nSamples, n_timewin, 4, 4, 128) ==>  (nSamples, n_timewin, 4*4*128)
            convnets = tf.reshape(convnets, shape=[-1, n_timewin, 4*4*128], name='Reshape_for_lstm')
            #lstm cell inputs:[batchs, time_steps, 4*4*128]
            with tf.variable_scope('LSTM_Cell'):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=1.0, state_is_tuple=True)
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, convnets, dtype=tf.float32, time_major=False)
                # outputs.shape is (batch_size, time_steps, num_units)
                outputs = tf.transpose(outputs, [1,0,2])        # (time_steps, batch_size, num_units)
                outputs = outputs[-1]

        with tf.variable_scope('Output_layers'):
            h_fc1_drop1 = tf.layers.dropout(outputs, rate=dropout_rate, training=train, name='dropout_1')
            h_fc1 = tf.layers.dense(h_fc1_drop1, 256, activation=tf.nn.relu, name='fc_relu_256')
            h_fc1_drop2 = tf.layers.dropout(h_fc1, rate=dropout_rate, training=train, name='dropout_2')
            prediction = tf.layers.dense(h_fc1_drop2, nb_classes, name='fc_softmax')

    return prediction


def build_convpool_mix(input_image, nb_classes, grad_clip=110, image_size=32, n_colors=3, 
        n_timewin=7, dropout_rate=0.5, num_units=128, batch_size=32, name='CNN_Mix', train=True, reuse=False):
    """
    Builds the complete network with LSTM and 1D-conv layers combined

    :param input_image: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param grad_clip:  the gradient messages are clipped to the given value during
                        the backward pass.
    :param imsize: size of the input image (assumes a square input)
    :param n_colors: number of color channels in the image
    :param n_timewin: number of time windows in the snippet
    :return: a pointer to the output of last layer
    """
    with tf.name_scope(name):
        with tf.name_scope('Parallel_CNNs'):
            convnets = []
            # Build 7 parallel CNNs with shared weights
            for i in range(n_timewin):
                if i==0:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=reuse)
                else:
                    convnet = build_cnn(input_image[i],image_size=image_size,n_colors=n_colors, reuse=True)
                convnets.append(convnet)
            convnets = tf.stack(convnets)
            convnets = tf.transpose(convnets, [1,0,2,3,4])

        with tf.variable_scope('Conv1d_over_flames'):
            convpool = tf.reshape(convnets, shape=[ -1, n_timewin, 4*4*128, 1])
            convpool = my_conv2d(convpool, filters=64, kernel_size=(3, 4*4*128), strides=(1, 1), padding='valid', activation=tf.nn.relu, name='convpool_conv1d')
            conv1d_out = tf.reshape(convpool, [-1, (n_timewin-2)*64])

        with tf.variable_scope('LSTM_layer'):
            # (nSamples, n_timewin, 4, 4, 128) ==>  (nSamples, n_timewin, 4*4*128)
            convnets = tf.reshape(convnets, shape=[-1, n_timewin, 4*4*128], name='Reshape_for_lstm')
            #lstm cell inputs:[batchs, time_steps, 4*4*128]
            with tf.variable_scope('LSTM_Cell'):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=1.0, state_is_tuple=True)
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, convnets, dtype=tf.float32, time_major=False)
                # outputs.shape is (batch_size, time_steps, num_units)
                outputs = tf.transpose(outputs, [1,0,2])
                lstm_out = outputs[-1]

        with tf.variable_scope('Output_layers'):
            dense_in = tf.concat((conv1d_out, lstm_out), axis=1, name='concat_conv1d_lstm')    # shape [batch, (n_timewin-2)*64+num_units]
            h_fc1_drop1 = tf.layers.dropout(dense_in, rate=dropout_rate, training=train, name='dropout_1')
            h_fc1 = tf.layers.dense(h_fc1_drop1, 512, activation=tf.nn.relu, name='fc_relu_512')
            h_fc1_drop2 = tf.layers.dropout(h_fc1, rate=dropout_rate, training=train, name='dropout_2')
            prediction = tf.layers.dense(h_fc1_drop2, nb_classes, name='fc_softmax')

    return prediction
