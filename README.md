# vgg16_quantization
There are two quantization results of tensorflow VGG16 model in INT8 and FP16 format. The repository provide some script that extract weight&bias(into *.npy) from tensorflow pb model and convert *.npy back to frozen tensorflow model for accurracy evaluation after quantization.
# Quantization solution
## INT8
### min-max scaling

## FP16
### cut-down conversion

# Result
The result is evaluated in ILSVRC2012 validation dataset. Hardware require 16GB at least.
## INT8
|     | IN8 quantization | FP16 quantization |
|:----|:----|:------|
|Accuracy Loss|<p>Top-1:0.91%<br>Top-5:0.78%</p>|<p>Top-1:0.028%<br>Top-5:0.022%</p>|

# Reference
[1][Tensorflow Model Garden](https://github.com/tensorflow/models)
