# vgg16_quantization
There are two quantization results of tensorflow VGG16 model in INT8 and FP16 format. The INT8 quantization implementation is edited from [1] say using bin quantization and FP16 is  .

# Quantization solution
## INT8
### bin quantization

### KL 

## FP16


# How to run
1. Install dependency
>pip install -r requirements.txt
2. Prepare Tensorflow Models
>Download tensorflow models from https://pan.baidu.com/s/1vrLqPi964cpxPO6OI9Wo7Q (password: 8vel) into current folder and unzip it (maybe faster) or clone from official repository[3]
3. Prepare float vgg16 tensorflow model
>Donload *.ckpt file from https://pan.baidu.com/s/1puRsX9FMV391oxrohVZVVQ (password: fvt5) into **checkpoints/** folder
4. INT8 Quantization
```python
python quantizer.py "checkpoints/vgg_16.ckpt" 1000 1 8
```
5. FP16 Quantization
```
python xxx.py
```

>Recommaded python3 version is under 3.8 because tensorflow 1.15 is not supported in higher version currently. 
>The code is tested in python3.7.3 on MacOS 10.15 and

# Result
The result is evaluated in ILSVRC2012 validation dataset. Hardware require 16GB at least.
## INT8
|     | bin quantization | KL quantization |
|:----|:----|:------|
|Accuracy Loss|<p>Top-1:<br>Top-5:</p>||
|Error(Avg)|2.21||

# Reference
[1][NNQuantization](https://github.com/bgrochal/NNQuantization)\
[2][Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim)\
[3][Tensorflow Model Garden](https://github.com/tensorflow/models)
