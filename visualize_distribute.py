import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

npy_file = "./vgg16.npy"

model_data = np.load(npy_file,allow_pickle=True).item()

useful_layer = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3',\
                'conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7','fc8']

for layer in useful_layer:
    x = model_data[layer]
    sns.displot(x)