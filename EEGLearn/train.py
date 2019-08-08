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

import os
import tensorflow as tf
import numpy as np
import scipy.io
import time
import datetime

from utils import reformatInput, load_or_generate_images, iterate_minibatches

from model import build_cnn, build_convpool_conv1d, build_convpool_lstm, build_convpool_mix


timestamp = datetime.datetime.now().strftime('%Y-%m-%d.%H.%M')
log_path = os.path.join("runs", timestamp)


model_type = '1dconv'      # ['1dconv', 'maxpool', 'lstm', 'mix', 'cnn']
log_path = log_path + '_' + model_type

batch_size = 32
dropout_rate = 0.5

input_shape = [32, 32, 3]   # 1024
nb_class = 4
n_colors = 3

# whether to train cnn first, and load its weight for multi-frame model
reuse_cnn_flag = False

# learning_rate for different models
lrs = {
    'cnn': 1e-3,
    '1dconv': 1e-4,
    'lstm': 1e-4,
    'mix': 1e-4,
}

weight_decay = 1e-4
learning_rate = lrs[model_type] / 32 * batch_size
optimizer = tf.train.AdamOptimizer

num_epochs = 60

def train(images, labels, fold, model_type, batch_size, num_epochs, subj_id=0, reuse_cnn=False, 
    dropout_rate=dropout_rate ,learning_rate_default=1e-3, Optimizer=tf.train.AdamOptimizer, log_path=log_path):
    """
    A sample training function which loops over the training set and evaluates the network
    on the validation set after each epoch. Evaluates the network on the training set
    whenever the
    :param images: input images
    :param labels: target labels
    :param fold: tuple of (train, test) index numbers
    :param model_type: model type ('cnn', '1dconv', 'lstm', 'mix')
    :param batch_size: batch size for training
    :param num_epochs: number of epochs of dataset to go over for training
    :param subj_id: the id of fold for storing log and the best model
    :param reuse_cnn: whether to train cnn first, and load its weight for multi-frame model
    :return: none
    """

    with tf.name_scope('Inputs'):
        input_var = tf.placeholder(tf.float32, [None, None, 32, 32, n_colors], name='X_inputs')
        target_var = tf.placeholder(tf.int64, [None], name='y_inputs')
        tf_is_training = tf.placeholder(tf.bool, None, name='is_training')

    num_classes = len(np.unique(labels))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)


    print('Train set label and proportion:\t', np.unique(y_train, return_counts=True))
    print('Val   set label and proportion:\t', np.unique(y_val, return_counts=True))
    print('Test  set label and proportion:\t', np.unique(y_test, return_counts=True))

    print('The shape of X_trian:\t', X_train.shape)
    print('The shape of X_val:\t', X_val.shape)
    print('The shape of X_test:\t', X_test.shape)
    

    print("Building model and compiling functions...")
    if model_type == '1dconv':
        network = build_convpool_conv1d(input_var, num_classes, train=tf_is_training, 
                            dropout_rate=dropout_rate, name='CNN_Conv1d'+'_sbj'+str(subj_id))
    elif model_type == 'lstm':
        network = build_convpool_lstm(input_var, num_classes, 100, train=tf_is_training, 
                            dropout_rate=dropout_rate, name='CNN_LSTM'+'_sbj'+str(subj_id))
    elif model_type == 'mix':
        network = build_convpool_mix(input_var, num_classes, 100, train=tf_is_training, 
                            dropout_rate=dropout_rate, name='CNN_Mix'+'_sbj'+str(subj_id))
    elif model_type == 'cnn':
        with tf.name_scope(name='CNN_layer'+'_fold'+str(subj_id)):
            network = build_cnn(input_var)  # output shape [None, 4, 4, 128]
            convpool_flat = tf.reshape(network, [-1, 4*4*128])
            h_fc1_drop1 = tf.layers.dropout(convpool_flat, rate=dropout_rate, training=tf_is_training, name='dropout_1')
            h_fc1 = tf.layers.dense(h_fc1_drop1, 256, activation=tf.nn.relu, name='fc_relu_256')
            h_fc1_drop2 = tf.layers.dropout(h_fc1, rate=dropout_rate, training=tf_is_training, name='dropout_2')
            network = tf.layers.dense(h_fc1_drop2, num_classes, name='fc_softmax')
            # the loss function contains the softmax activation
    else:
        raise ValueError("Model not supported ['1dconv', 'maxpool', 'lstm', 'mix', 'cnn']")

    Train_vars = tf.trainable_variables()

    prediction = network

    with tf.name_scope('Loss'):
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in Train_vars if 'kernel' in v.name])
        ce_loss = tf.losses.sparse_softmax_cross_entropy(labels=target_var, logits=prediction)
        _loss = ce_loss + weight_decay*l2_loss

    # decay_steps learning rate decay
    decay_steps = 3*(len(y_train)//batch_size)   # len(X_train)//batch_size  the training steps for an epcoh
    with tf.name_scope('Optimizer'):
        # learning_rate = learning_rate_default * Decay_rate^(global_steps/decay_steps)
        global_steps = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(     # learning rate decay
            learning_rate_default,  # Base learning rate.
            global_steps,
            decay_steps,
            0.95,  # Decay rate.
            staircase=True)
        optimizer = Optimizer(learning_rate)    # GradientDescentOptimizer  AdamOptimizer
        train_op = optimizer.minimize(_loss, global_step=global_steps, var_list=Train_vars)

    with tf.name_scope('Accuracy'):
        prediction = tf.argmax(prediction, axis=1)
        correct_prediction = tf.equal(prediction, target_var)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # Output directory for models and summaries
    # choose different path for different model and subject
    out_dir = os.path.abspath(os.path.join(os.path.curdir, log_path, (model_type+'_'+str(subj_id)) ))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss, accuracy and learning_rate
    loss_summary = tf.summary.scalar('loss', _loss)
    acc_summary = tf.summary.scalar('train_acc', accuracy)
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, lr_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, tf.get_default_graph())

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, tf.get_default_graph())

    # Test summaries
    test_summary_op = tf.summary.merge([loss_summary, acc_summary])
    test_summary_dir = os.path.join(out_dir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, tf.get_default_graph())


    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, model_type)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    if model_type != 'cnn' and reuse_cnn:
        # saver for reuse the CNN weight
        reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='VGG_NET_CNN')
        original_saver = tf.train.Saver(reuse_vars)         # Pass the variables as a list

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    print("Starting training...")
    total_start_time = time.time()
    best_validation_accu = 0

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        if model_type != 'cnn' and reuse_cnn:
            cnn_model_path = os.path.abspath(
                                os.path.join(
                                    os.path.curdir, log_path, ('cnn_'+str(subj_id)), 'checkpoints' ))
            cnn_model_path = tf.train.latest_checkpoint(cnn_model_path)
            print('-'*20)
            print('Load cnn model weight for multi-frame model from {}'.format(cnn_model_path))
            original_saver.restore(sess, cnn_model_path)

        stop_count = 0  # count for earlystopping
        for epoch in range(num_epochs):
            print('-'*50)
            # Train set
            train_err = train_acc = train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets = batch
                summary, _, pred, loss, acc = sess.run([train_summary_op, train_op, prediction, _loss, accuracy], 
                    {input_var: inputs, target_var: targets, tf_is_training: True})
                train_acc += acc
                train_err += loss
                train_batches += 1
                train_summary_writer.add_summary(summary, sess.run(global_steps))

            av_train_err = train_err / train_batches
            av_train_acc = train_acc / train_batches

            # Val set
            summary, pred, av_val_err, av_val_acc = sess.run([dev_summary_op, prediction, _loss, accuracy],
                    {input_var: X_val, target_var: y_val, tf_is_training: False})
            dev_summary_writer.add_summary(summary, sess.run(global_steps))

            
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            
            fmt_str = "Train \tEpoch [{:d}/{:d}]  train_Loss: {:.4f}\ttrain_Acc: {:.2f}"
            print_str = fmt_str.format(epoch + 1, num_epochs, av_train_err, av_train_acc*100)
            print(print_str)

            fmt_str = "Val \tEpoch [{:d}/{:d}]  val_Loss: {:.4f}\tval_Acc: {:.2f}"
            print_str = fmt_str.format(epoch + 1, num_epochs, av_val_err, av_val_acc*100)
            print(print_str)
            
            # Test set
            summary, pred, av_test_err, av_test_acc = sess.run([test_summary_op, prediction, _loss, accuracy],
                {input_var: X_test, target_var: y_test, tf_is_training: False})
            test_summary_writer.add_summary(summary, sess.run(global_steps))
            
            fmt_str = "Test \tEpoch [{:d}/{:d}]  test_Loss: {:.4f}\ttest_Acc: {:.2f}"
            print_str = fmt_str.format(epoch + 1, num_epochs, av_test_err, av_test_acc*100)
            print(print_str)

            if av_val_acc > best_validation_accu:   # early_stoping
                stop_count = 0
                eraly_stoping_epoch = epoch
                best_validation_accu = av_val_acc
                test_acc_val = av_test_acc
                saver.save(sess, checkpoint_prefix, global_step=sess.run(global_steps))
            else:
                stop_count += 1
                if stop_count >= 10: # stop training if val_acc dose not imporve for over 10 epochs
                    break

        train_batches = train_acc = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets = batch
            acc = sess.run(accuracy, {input_var: X_train, target_var: y_train, tf_is_training: False})
            train_acc += acc
            train_batches += 1

        last_train_acc = train_acc / train_batches
        
        
        last_val_acc = av_val_acc
        last_test_acc = av_test_acc
        print('-'*50)
        print('Time in total:', time.time()-total_start_time)
        print("Best validation accuracy:\t\t{:.2f} %".format(best_validation_accu * 100))
        print("Test accuracy when got the best validation accuracy:\t\t{:.2f} %".format(test_acc_val * 100))
        print('-'*50)
        print("Last train accuracy:\t\t{:.2f} %".format(last_train_acc * 100))
        print("Last validation accuracy:\t\t{:.2f} %".format(last_val_acc * 100))
        print("Last test accuracy:\t\t\t\t{:.2f} %".format(last_test_acc * 100))
        print('Early Stopping at epoch: {}'.format(eraly_stoping_epoch+1))

    train_summary_writer.close()
    dev_summary_writer.close()
    test_summary_writer.close()
    return [last_train_acc, best_validation_accu, test_acc_val, last_val_acc, last_test_acc]



def train_all_model(num_epochs=3000):
    nums_subject = 13
    # Leave-Subject-Out cross validation
    subj_nums = np.squeeze(scipy.io.loadmat('../SampleData/trials_subNums.mat')['subjectNum'])
    fold_pairs = []
    for i in np.unique(subj_nums):
        ts = subj_nums == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        np.random.shuffle(tr)
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))


    images_average, images_timewin, labels = load_or_generate_images(
                                                file_path='../SampleData/', average_image=3)


    print('*'*200)
    acc_buf = []
    for subj_id in range(nums_subject):
        print('-'*100)
        
        if model_type == 'cnn':
            print('The subjects', subj_id, '\t\t Training the ' + 'cnn' + ' Model...')
            acc_temp = train(images_average, labels, fold_pairs[subj_id], 'cnn', 
                                batch_size=batch_size, num_epochs=num_epochs, subj_id=subj_id,
                                learning_rate_default=lrs['cnn'], Optimizer=optimizer, log_path=log_path)
            acc_buf.append(acc_temp)
            tf.reset_default_graph()
            print('Done!')

        else:
            # whether to train cnn first, and load its weight for multi-frame model
            if reuse_cnn_flag is True:
                print('The subjects', subj_id, '\t\t Training the ' + 'cnn' + ' Model...')
                acc_temp = train(images_average, labels, fold_pairs[subj_id], 'cnn', 
                                    batch_size=batch_size, num_epochs=num_epochs, subj_id=subj_id,
                                    learning_rate_default=lrs['cnn'], Optimizer=optimizer, log_path=log_path)
                # acc_buf.append(acc_temp)
                tf.reset_default_graph()
                print('Done!')
        
            print('The subjects', subj_id, '\t\t Training the ' + model_type + ' Model...')
            print('Load the CNN model weight for backbone...')
            acc_temp = train(images_timewin, labels, fold_pairs[subj_id], model_type, 
                            batch_size=batch_size, num_epochs=num_epochs, subj_id=subj_id, reuse_cnn=reuse_cnn_flag, 
                            learning_rate_default=learning_rate, Optimizer=optimizer, log_path=log_path)
                                
            acc_buf.append(acc_temp)
            tf.reset_default_graph()
            print('Done!')
        
        # return

    print('All folds for {} are done!'.format(model_type))
    acc_buf = (np.array(acc_buf)).T
    acc_mean = np.mean(acc_buf, axis=1).reshape(-1, 1)
    acc_buf = np.concatenate([acc_buf, acc_mean], axis=1)
    # the last column is the mean of current row
    print('Last_train_acc:\t', acc_buf[0], '\tmean :', np.mean(acc_buf[0][-1]))
    print('Best_val_acc:\t', acc_buf[1], '\tmean :', np.mean(acc_buf[1][-1]))
    print('Earlystopping_test_acc:\t', acc_buf[2], '\tmean :', np.mean(acc_buf[2][-1]))
    print('Last_val_acc:\t', acc_buf[3], '\tmean :', np.mean(acc_buf[3][-1]))
    print('Last_test_acc:\t', acc_buf[4], '\tmean :', np.mean(acc_buf[4][-1]))
    np.savetxt('./Accuracy_{}.csv'.format(model_type), acc_buf, fmt='%.4f', delimiter=',')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    np.random.seed(2018)
    tf.set_random_seed(2018)

    train_all_model(num_epochs=num_epochs)
