# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 17:40
# @Author  : zhoujun

# data config
trainroot = '/home/gp/work/project/PSENet.pytorch.resnet18/data/jizhuangxiang/train'
testroot = '/home/gp/work/project/PSENet.pytorch.resnet18/data/jizhuangxiang/test'
output_dir = 'output/psenet_jizhuangxiang_resnet18_1gpu_author_crop_adam_MultiStepLR_authorloss'
data_shape = 640

# train config
gpu_id = '0'
workers = 4
start_epoch = 0
epochs = 100

train_batch_size = 4

lr = 1e-4
end_lr = 1e-7
lr_gamma = 0.1
lr_decay_step = [30,60]
weight_decay = 5e-4
warm_up_epoch = 6
warm_up_lr = lr * lr_gamma

display_input_images = False
display_output_images = False
display_interval = 10
show_images_interval = 50

pretrained = True
restart_training = True
checkpoint = ''

# net config
backbone = 'resnet18'   # 'resnet18', 'shufflenetv2', 'MobileNetV3_Small'
Lambda = 0.7
n = 6
m = 0.5
OHEM_ratio = 3
scale = 1
# random seed
seed = 2


def print():
    from pprint import pformat
    tem_d = {}
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            tem_d[k] = v
    return pformat(tem_d)
