# coding: utf-8

import os

f_path = os.path.abspath('..')
# print(f_path.split('shadow_code'))
root_path = f_path.split('shadow_code')[0]
ViSha_training_root = (root_path+'datasets/shadow/train','video','ViSD_train')
ViSha_validation_root = (root_path+'datasets/shadow/test', 'video', 'ViSD_test')