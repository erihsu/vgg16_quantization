# -*- coding: utf-8 -*-
# @Author: Eric Xu
# @Date:   2020-06-26 23:02:33
# @Last Modified by:   Eric Xu
# @Last Modified time: 2020-06-27 10:06:25


# details: https://www.easck.com/cos/2020/0622/549363.shtml

#coding=gbk
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


checkpoint_path='./float/vgg_16.ckpt'#your ckpt path

reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()

vgg16={}
vgg16_layer = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7','fc8']
add_info = ['weights','biases']

vgg16={'conv1_1':[[],[]],'conv1_2':[[],[]],'conv2_1':[[],[]],'conv2_2':[[],[]],'conv3_1':[[],[]],'conv3_2':[[],[]], \
       'conv3_3':[[],[]],'conv4_1':[[],[]],'conv4_2':[[],[]],'conv4_3':[[],[]],'conv5_1':[[],[]],'conv5_2':[[],[]], \
       'conv5_3':[[],[]],'fc6':[[],[]],'fc7':[[],[]],'fc8':[[],[]]}

for key in var_to_shape_map:
    str_name = key
    print(str_name)
 # 因为模型使用Adam算法优化的，在生成的ckpt中，有Adam后缀的tensor
    if str_name.find('Adam') > -1:
        continue
    if str_name.find('/') > -1:
        names = str_name.split('/')
        new_name = " ".join(names)
        if "fc" in new_name:
            layer_name = names[1]
            layer_add_info = names[2]
        elif len(names) >= 4:
            layer_name = names[2]
            layer_add_info = names[3]
        else: 
            layer_name = ""
            layer_add_info = ""
    else:
        layer_name = str_name
        layer_add_info = None

    if layer_add_info == 'weights':
        vgg16[layer_name][0]=reader.get_tensor(key)
        print(layer_name + str(reader.get_tensor(key).shape))
    elif layer_add_info == 'biases':
        print(layer_name + str(reader.get_tensor(key).shape))
        vgg16[layer_name][1] = reader.get_tensor(key)
    else:
        vgg16[layer_name] = reader.get_tensor(key)


# save npy
np.save('vgg16.npy',vgg16,allow_pickle=True)
print('save npy over...')
print(vgg16['fc8'][0].shape)
print(vgg16['fc8'][1].shape)
