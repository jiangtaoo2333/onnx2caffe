# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-10-25 14:01:38
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2021-11-03 10:10:33
'''
import argparse
import glob
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import copy

sys.path.append("/jiangtao2/caffe-master/python/")
dirpath = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(dirpath)

import caffe


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')

def parse_args():

   parser = argparse.ArgumentParser(description='test caffe model from pytorch')

   parser.add_argument('--model', dest='model_proto', 
                        default='./model/model_best_sim.prototxt', type=str)
   parser.add_argument('--weights', dest='model_weight', 
                        default=['./model/model_best_sim.caffemodel',], type=list)
   parser.add_argument('--testsize', dest='testsize', 
                        default=192,type=int)

   args = parser.parse_args()
   return args

def start_test(model_proto,model_weight,testsize):

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(model_proto, model_weight, caffe.TEST)

    # 设置图片的格式
    # img = np.zeros((192,192,3))
    # img = img * 0.0039216
    # img = img[:,:,np.newaxis]

    # 前向传播
    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # transformer.set_transpose('data', (2,0,1))
    # result = net.forward_all(data=np.asarray([transformer.preprocess('data', img)]))
    # result = net.forward_all(input=np.zeros((1,3,192,192)))
    # print(result)

    # hotmap = result['730']
    # gaze = result['771']

    # print(hotmap.shape)
    # print(gaze.shape)
    # print(hotmap[0][0][0][0])
    # print(gaze)

    # for i in range(800):
    #     print(i)
    image = cv2.imread('./image/0.jpg',0)
    image = image[:,:,np.newaxis]
    image = image * 0.0039216
    image = np.transpose(image,(2,0,1))
    image = image[np.newaxis]

    # image = np.zeros((1,1,192,192))

    net.blobs['input'].data[...] = image
    out = net.forward()

    # res = net.blobs['521'].data[0]
    # print(res.shape)
    # res = net.blobs['521R'].data[0]
    # print(res.shape)
    # print(res)

    res = net.blobs['694'].data
    print(res.shape)
    # print(res[0][0])

    res = net.blobs['694R'].data
    print(res.shape)
    res0 = res[0][0]
    print(res0.shape)
    print(res0.max())
    # print(type(res))
    # print(res[0][0])
    # print(res[0])
    # print(res[0][1][0])
    # res = net.blobs['735'].data[0]
    # print(res[0])
    # print(res[1])


    with open('./res/caffe_res.txt','a') as f:
        for i in range(1):
            for j in range(17):
                for m in range(48):
                    for n in range(48):
                        f.write(str(res[i][j][m][n])+' ')
                        f.write('\n')

    # filters = net.params['Conv_0'][0].data
    # print(filters.shape)





if __name__ == '__main__':

    args = parse_args()
    
    for modelWeight in args.model_weight:
        start_test(args.model_proto,modelWeight,args.testsize)
