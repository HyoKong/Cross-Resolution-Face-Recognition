from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import scipy.misc as m
import os
import random

# img_path = '../00003.jpg'
# img = Image.open(img_path)
# img = img.convert('RGB')

# img = img.convert('YCbCr')
crop_size = 112
reference = get_reference_facial_points(default_square = True) * crop_size / 112.0

TARGET_ROOT = '/media/hyo/文档/VIS-NIR/SKETCH/cropped_sketch'
SOURCE_ROOT_LIST = ['/media/hyo/文档/VIS-NIR/SKETCH/CUFS/AR_sketch/sketch',
                    '/media/hyo/文档/VIS-NIR/SKETCH/CUFS/CUHK_testing_sketch/sketch',
                    '/media/hyo/文档/VIS-NIR/SKETCH/CUFS/CUHK_training_sketch/sketch',
                    '/media/hyo/文档/VIS-NIR/SKETCH/CUFS/XM2VTS_sketch/sketch',
                    '/media/hyo/文档/VIS-NIR/SKETCH/CUFSF/original_sketch']
count = 0
random.shuffle(SOURCE_ROOT_LIST)
for source_root in SOURCE_ROOT_LIST:
    img_list = os.listdir(source_root)
    for img_name in img_list:
        img = Image.open(os.path.join(source_root,img_name))
        img = img.convert('RGB')
        _ , landmark = detect_faces(img)
        if len(landmark) == 0:
            print('abcdefghijklmnopqrstuvwxyz')
            continue
        else:
            count += 1
            target_name = str(count).zfill(5) + '.jpg'
            facial5points = [[landmark[0][j],landmark[0][j+5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face[0])
            print(target_name)
            # print(facial_5_points)
            # left_eye_landmark = facial_5_points[0]
            # img = img.crop((left_eye_landmark[0]-11,left_eye_landmark[1]-11,left_eye_landmark[0]+11,left_eye_landmark[1]+11))
            
            # imgArr = np.array(img)
            # imgYArr = imgArr.copy()
            # plt.imshow(img_warped)
            # plt.show()
            # image.show()
            img_warped.save(os.path.join(TARGET_ROOT,target_name),quality=100)
