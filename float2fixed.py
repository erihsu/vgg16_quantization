# Ref:

import numpy as np

npy_file = "./vgg16.npy"
model_data = np.load(npy_file,allow_pickle=True).item()
useful_layer = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3',\
                'conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7','fc8']
vgg16_int={'conv1_1':[[],[]],'conv1_2':[[],[]],'conv2_1':[[],[]],'conv2_2':[[],[]],'conv3_1':[[],[]],'conv3_2':[[],[]], \
       'conv3_3':[[],[]],'conv4_1':[[],[]],'conv4_2':[[],[]],'conv4_3':[[],[]],'conv5_1':[[],[]],'conv5_2':[[],[]], \
       'conv5_3':[[],[]],'fc6':[[],[]],'fc7':[[],[]],'fc8':[[],[]]}
vgg16_fp={'conv1_1':[[],[]],'conv1_2':[[],[]],'conv2_1':[[],[]],'conv2_2':[[],[]],'conv3_1':[[],[]],'conv3_2':[[],[]], \
       'conv3_3':[[],[]],'conv4_1':[[],[]],'conv4_2':[[],[]],'conv4_3':[[],[]],'conv5_1':[[],[]],'conv5_2':[[],[]], \
       'conv5_3':[[],[]],'fc6':[[],[]],'fc7':[[],[]],'fc8':[[],[]]}

def weight_quantize(data):
    sign = 0
    if data == 0:
        return 0
    elif data > 0:
        sign = 0
        data = data/0.6 # scalor
        data_tmp = round(data*128)
        return data_tmp/128
    else:
        sign = 1
        data = -data/0.6 # scalor
        data_tmp = round(data*128)
        return -data_tmp/128


def float2fixed(z):
    new_func = np.vectorize(weight_quantize)
    return new_func(z)

def float2halffp(data):
    data = data.astype(np.float16)
    data = data.astype(np.float32)
    return data

for layer in useful_layer:
    vgg16_fp[layer][0] = float2halffp(model_data[layer][0]) # convert weights(fp16)
    vgg16_fp[layer][1] = float2halffp(model_data[layer][1])  # convert bias(fp16)
    vgg16_int[layer][0] = float2fixed(model_data[layer][0]).astype(np.float32) # convert weights(int8)
    vgg16_int[layer][1] = np.zeros(model_data[layer][1].shape,dtype=np.float32) # convert bias(int8)
# save npy
np.save('vgg16_int8.npy',vgg16_int,allow_pickle=True)
np.save('vgg16_fp.npy',vgg16_fp,allow_pickle=True)
print('save npy over...')
