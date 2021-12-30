# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-10-26 11:20:03
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2021-11-03 10:04:46
'''
import argparse
import os
import sys
import time
import uuid
import xml.dom.minidom
from xml.dom.minidom import Document
import onnx
import onnxruntime
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import copy
from collections import OrderedDict


def testOnnxSim(image):
    # 读取onnx模型
    onnxPath = './model/model_best_sim.onnx'
    session = onnxruntime.InferenceSession(onnxPath)

    # 读取输入输出节点的名字
    input_name = session.get_inputs()[0].name
    label_name_0 = session.get_outputs()[0].name
    label_name_1 = session.get_outputs()[1].name

    # 对输入的维度进行更改，主要是增加第一维度，即为Batchsize
    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze_(0)

    # 前向传播
    result = session.run([label_name_0,label_name_1], {input_name: tensor.cpu().numpy()})

    return result

def testOnnxSimFull(image):

    # 读取onnx模型
    onnxPath = './model/model_best_sim.onnx'
    model = onnx.load(onnxPath)
    ori_output = copy.deepcopy(model.graph.output)
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    session = onnxruntime.InferenceSession(model.SerializeToString())


    # 读取输入输出节点的名字
    outputs = [x.name for x in session.get_outputs()]
    outputs = outputs[2:] #去除重复添加的两个节点
    input_name = session.get_inputs()[0].name
    label_name_0 = session.get_outputs()[0].name
    label_name_1 = session.get_outputs()[1].name

    # 对输入的维度进行更改，主要是增加第一维度，即为Batchsize
    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze_(0)

    # 前向传播
    result = session.run(outputs, {input_name: tensor.cpu().numpy()})

    return result,outputs

if __name__ == '__main__':

    # image = np.zeros((1,192,192))
    # result,outputs = testOnnxSimFull(image)
    # ort_outs = OrderedDict(zip(outputs, result))

    # print(ort_outs['694'].shape)
    # print(ort_outs['694'][0][0][0][0])
    # print(ort_outs['694'][0][0][1][0])

    # print(ort_outs['735'].shape)
    # print(ort_outs['735'][0][0])
    # print(ort_outs['735'][0][1])

    # for i in range(800):
    #     print(i)
    image = cv2.imread('./image/0.jpg',0)
    image = image[:,:,np.newaxis]
    image = np.transpose(image,(2,0,1))
    image = image * 0.0039216
    # image = np.zeros((1,192,192))
    result,outputs = testOnnxSimFull(image)
    ort_outs = OrderedDict(zip(outputs, result))

    print(ort_outs['694'].shape)
    print(ort_outs['694'][0][0][0][0])
    print(ort_outs['694'][0][0][1][0])

    print(ort_outs['735'].shape)
    print(ort_outs['735'][0][0])
    print(ort_outs['735'][0][1])

    with open('./res/onnx_res.txt','a') as f:
        for i in range(1):
            for j in range(17):
                for m in range(48):
                    for n in range(48):
                        f.write(str(ort_outs['694'][i][j][m][n])+' ')
                        f.write('\n')