# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-10-27 11:02:05
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2021-11-01 15:45:19
'''
import cv2
import numpy as np
import skimage
import skimage.transform
import torch
import sys
sys.path.append("/jiangtao2/caffe-master/python/")
import os
os.environ['GLOG_minloglevel'] = '3' #关闭caffe打印
import caffe
import numpy as np
from caffe import layers as L
from caffe.proto import caffe_pb2
import skimage


def create_upsample_net(shape, factor):
    kernel_size = 2 * factor - factor % 2
    # pad = factor // 2
    pad = int(np.ceil((factor - 1) / 2.))
    stride = factor
    num_output = shape[1]
    group = shape[1]
    
    blob_shape = caffe_pb2.BlobShape()
    blob_shape.dim.extend(shape)
    n = caffe.NetSpec()
    n.data = L.Input(shape=[blob_shape])
    convolution_param = dict(num_output=num_output, 
                             kernel_size=kernel_size, 
                             stride=stride, 
                             group=num_output,
                             pad=pad,
                             weight_filler=dict(type="bilinear"),
                             bias_term=False)
    n.conv = L.Deconvolution(n.data, 
                             convolution_param=convolution_param,
                             param=dict(lr_mult=0, decay_mult=0))

    prototxt = 'temp.prototxt'
    with open(prototxt, 'w') as f:
        f.write(str(n.to_proto()))
    net = caffe.Net(prototxt, caffe.TEST)
    # os.remove(prototxt)
    return net

def test_caffe_upsample(data, factor):
    net = create_upsample_net(data.shape, factor)
    net.blobs['data'].data[...] = data
    net.forward()
    caffe_output = net.blobs['conv'].data

    x = np.squeeze(data, axis=0)
    x = np.transpose(x, (1,2,0))
    skimage_output = skimage.transform.rescale(x, factor, mode='constant', cval=0)
    skimage_output = np.transpose(skimage_output, (2,0,1))
    skimage_output = np.expand_dims(skimage_output, axis=0)

    print('caffe_output[0][0]',caffe_output[0][0])
    print('skimage_output[0][0]',skimage_output[0][0])

def test_pytorch_upsample(data, factor):
    x = np.squeeze(data, axis=0)
    x = np.transpose(x, (1,2,0))
    skimage_output = skimage.transform.rescale(x, factor, mode='constant', 
                                               cval=0, multichannel=True)
    skimage_output = np.transpose(skimage_output, (2,0,1))
    skimage_output = np.expand_dims(skimage_output, axis=0)

    upsample = torch.nn.UpsamplingNearest2d(scale_factor=factor)
    torch_output = upsample(torch.from_numpy(data)).numpy()

    upsample_2 = torch.nn.Upsample(scale_factor=factor, mode='bilinear', 
                                   align_corners=True)
    torch_output_2 = upsample_2(torch.from_numpy(data)).numpy()


    print('skimage_output[0][0]',skimage_output[0][0])
    print('torch_output[0][0]',torch_output[0][0])
    print('torch_output_2[0][0]',torch_output_2[0][0])

if __name__ == '__main__':

    with open('./res/caffe_res.txt','r') as f:
        caffelines = f.readlines()
    with open('./res/nov_res.txt','r') as f:
        novlines = f.readlines()

    cafferes = np.zeros((800,2))
    for line in caffelines:
        line = line.strip()
        line = line.split(' ')
        idx = int(line[0])
        gazex = float(line[1])
        gazey = float(line[2])
        cafferes[idx][0] = gazex
        cafferes[idx][1] = gazey

    novres = np.zeros((800,2))
    for line in novlines:
        line = line.strip()
        line = line.split(' ')
        idx = int(line[0])
        gazex = float(line[1])
        gazey = float(line[2])
        novres[idx][0] = gazex
        novres[idx][1] = gazey
    
    minus = cafferes - novres
    minus = abs(minus)
    print(minus.mean(axis=0))
    print(minus.mean())