# -*- coding: utf-8 -*-
# @Author: Eric Xu
# @Date:   2020-06-27 09:24:16
# @Last Modified by:   Eric Xu
# @Last Modified time: 2020-06-27 11:05:58

# Ref: https://github.com/LiMingda92/VGG16_TF

import tensorflow as tf
import numpy as np 

data_dict = np.load('./vgg16_int8.npy',allow_pickle=True).item()

def conv(x, d_out, name):
    d_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.constant(data_dict[name][0], name="weights")
        bias = tf.constant(data_dict[name][1], name="bias")
        conv = tf.nn.conv2d(x, kernel,[1, 1, 1, 1], padding='SAME')
        BiasAdd = tf.nn.bias_add(conv,bias)
        activation = tf.nn.relu(BiasAdd, name=scope)
        return activation

def maxpool(x, name):
    activation = tf.nn.max_pool2d(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name)
    return activation

def fc(x, n_out, name):
    d_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight = tf.constant(data_dict[name][0], name="weights")
        bias = tf.constant(data_dict[name][1], name="bias")
        useless_var = tf.Variable(10,name="nouse")
        conv = tf.nn.conv2d(x, weight,[1, 1, 1, 1], padding='VALID')
        BiasAdd = tf.nn.bias_add(conv,bias)
        activation = tf.nn.relu(BiasAdd, name=scope)
        # activation = tf.nn.relu_layer(x, weight, bias, name=name)
        return activation

def fc_last(x,name):
    with tf.name_scope(name) as scope:
        weight = tf.constant(data_dict[name][0], name="weights")
        bias = tf.constant(data_dict[name][1], name="bias")
        conv = tf.nn.conv2d(x, weight,[1, 1, 1, 1], padding='VALID')
        BiasAdd = tf.nn.bias_add(conv, bias)
        sq = tf.squeeze(BiasAdd,squeeze_dims=[1,2])
        return sq

def VGG16(images, n_cls):

    conv1_1 = conv(images, 64, 'conv1_1')
    conv1_2 = conv(conv1_1, 64, 'conv1_2')
    pool1   = maxpool(conv1_2, 'pool1')

    conv2_1 = conv(pool1, 128, 'conv2_1')
    conv2_2 = conv(conv2_1, 128, 'conv2_2')
    pool2   = maxpool(conv2_2, 'pool2')

    conv3_1 = conv(pool2, 256, 'conv3_1')
    conv3_2 = conv(conv3_1, 256, 'conv3_2')
    conv3_3 = conv(conv3_2, 256, 'conv3_3')
    pool3   = maxpool(conv3_3, 'pool3')

    conv4_1 = conv(pool3, 512, 'conv4_1')
    conv4_2 = conv(conv4_1, 512, 'conv4_2')
    conv4_3 = conv(conv4_2, 512, 'conv4_3')
    pool4   = maxpool(conv4_3, 'pool4')

    conv5_1 = conv(pool4, 512, 'conv5_1')
    conv5_2 = conv(conv5_1, 512, 'conv5_2')
    conv5_3 = conv(conv5_2, 512, 'conv5_3')
    pool5   = maxpool(conv5_3, 'pool5')

    '''
    因为训练自己的数据，全连接层最好不要使用预训练参数
    '''
    # flatten  = tf.reshape(pool5, [-1, 7*7*512])
    fc6      = fc(pool5, 4096, 'fc6')
    # dropout1 = tf.nn.dropout(fc6, _dropout)

    fc7      = fc(fc6, 4096, 'fc7')
    # dropout2 = tf.nn.dropout(fc7, _dropout)
    
    fc8      = fc_last(fc7, 'fc8')

    return fc8


def test(path):
    x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    # keep_prob = tf.placeholder(tf.float32)
    output = VGG16(x, 1000)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, './vgg.ckpt-999/vgg.ckpt-999')

if __name__ == '__main__':
    path = '.'
    test(path)