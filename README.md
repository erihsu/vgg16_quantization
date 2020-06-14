# vgg16_quantization
There are two quantization results of vgg16 in INT8 and FP16 format. The INT8 quantization implementation is edited from [1] say using bin quantization and FP16 is  .

# Quantization solution
## INT8
### bin quantization

### KL 

## FP16


# How to run
1. Install dependency
>pip install -r requirements.txt
2. INT8 Quantization
```python
python quantizer.py "checkpoints/vgg_16.ckpt" 1000 1 8
```
3. FP16 Quantization
```
python xxx.py
```

>Recommaded python3 version is under 3.8 because tensorflow 1.15 is not supported in higher version. 
>The code is tested in python3.7.3 on MacOS 10.15 and

# Reference
[1][NNQuantization](https://github.com/bgrochal/NNQuantization)\
[2][Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim)\
[3][Tensorflow Model Garden](https://github.com/tensorflow/models)
