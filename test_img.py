# -*- coding: utf-8 -*-
# @Time    : 1/4/19 11:14 AM
# @Author  : zhoujun
import torch
from torchvision import transforms
import os
import cv2
import time
import random
import numpy as np

import config
from models import PSENet
import matplotlib.pyplot as plt
from utils.utils import show_img, draw_bbox

from pse import decode as pse_decode
from tqdm import tqdm


class Pytorch_model:
    def __init__(self, model_path, net, scale, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.scale = scale
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")
        self.net = torch.load(model_path, map_location=self.device)['state_dict']
        print('device:', self.device)

        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            net.scale = scale
            try:
                sk = {}
                for k in self.net:
                    sk[k[7:]] = self.net[k]
                net.load_state_dict(sk)
            except:
                net.load_state_dict(self.net)
            self.net = net
            print('load models')
        self.net.eval()

    def predict(self, img: str, long_size: int = 2240):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img), 'file is not exists'
        img = cv2.imread(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            preds = self.net(tensor)
            preds, boxes_list = pse_decode(preds[0], self.scale)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            # print(scale)
            # preds, boxes_list = decode(preds,num_pred=-1)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            torch.cuda.synchronize()
            t = time.time() - start
        return preds, boxes_list, t


def get_image_list(image_dir, suffix=['jpg', 'jpeg', 'JPG', 'JPEG','png']):
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        print("PATH:%s not exists" % image_dir)
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath)
    return imglist



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    model_path = 'output/psenet_jizhuangxiang_resnet18_1gpu_author_crop_adam_MultiStepLR_authorloss/PSENet_0_loss0.141679.pth'
    img_dir = '/home/gp/work/project/PSENet.pytorch.resnet18/data/jizhuangxiang/train/img'
    result_dir = 'test_result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_list = get_image_list(img_dir)
    img_list = random.sample(img_list, 50)

    # 初始化网络
    net = PSENet(backbone='resnet18', pretrained=False, result_num=config.n)
    model = Pytorch_model(model_path, net=net, scale=1, gpu_id=0)
    for img_path in tqdm(img_list):
        preds, boxes_list,t = model.predict(img_path)

        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        cv2.imwrite(os.path.join(result_dir, os.path.basename(img_path)), img)

