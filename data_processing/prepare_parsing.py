from PIL import Image
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm
import argparse
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import math

if __name__ == "__main__":
    
    source_root = '/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original/labels (copy)/'
    target_root = '/media/hyo/文档/Dataset/SmithCVPR2013_dataset_original/labels_aligned_renew_v2/'
    helen_root = '/media/hyo/文档/Dataset/Helen/'
    # 放大倍数
    crop_size = 224
    scale = crop_size / 112.
    # scale = crop_size / 224. *1.28
    original_5_points = get_reference_facial_points(default_square=True)
    original_5_points[:, 1] = original_5_points[:, 1]
    reference = (original_5_points) * scale
    
    # reference = (get_reference_facial_points(default_square = True)+15) * scale
    cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(target_root):
        os.mkdir(target_root)
    
    
    # 获得旋转后图片的名称
    def rotate_img_name(img_name,angle):
        temp = img_name.split('.')
        angle = str(angle)
        if 'r' in img_name:
            new_img_name = temp[0]+'_'+angle+'.jpg'
        else:
            new_img_name = temp[0]+'_'+angle+'_r.jpg'
        return new_img_name
    
    
    def rotate_matrix(angle):
        # 角度变弧度
        angle = angle/180*math.pi
        matrix = [[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]]
        return matrix
    
    df = pd.read_csv('/home/hyo/original_landmarks.csv')
    
    sub_list = os.listdir(helen_root)
    image_list = []
    for subfolder in sub_list:
        list = os.path.join(helen_root,subfolder)
        for image in os.listdir(list):
            image_list.append(image.split('.')[0])
            
    
    # for subfolder in os.listdir(source_root):
    #     image_name = subfolder          # 例如：11564757_1
        # 对原始parsing map进行crop处理
    for subfolder in image_list:
        image_name = subfolder
        for lbl in os.listdir(os.path.join(source_root,subfolder)):
            print("Processing\t{}".format(os.path.join(source_root, subfolder, lbl)))
            # 读取每一个label图
            img = Image.open(os.path.join(source_root, subfolder, lbl))
            # 根据ground truth landmark获取5个关键点
            facial5points = []
            gt_landmarks = df[df['img_name'] == image_name+'.jpg'].values.flatten()[1:]
            gt_landmarks_array = gt_landmarks[:].reshape(-1, 2)
            # pdb.set_trace()

            facial5points.append([np.average(gt_landmarks_array[134:154, :].tolist(), axis=0)[0],
                                  np.average(gt_landmarks_array[134:154, :].tolist(), axis=0)[1]])
            facial5points.append([np.average(gt_landmarks_array[114:134, :].tolist(), axis=0)[0],
                                  np.average(gt_landmarks_array[114:134, :].tolist(), axis=0)[1]])
            facial5points.append([np.average(gt_landmarks_array[41:58, :].tolist(), axis=0)[0],
                                  np.average(gt_landmarks_array[41:58, :].tolist(), axis=0)[1]])
            facial5points.append([np.average(gt_landmarks_array[58:59, :].tolist(), axis=0)[0],
                                  np.average(gt_landmarks_array[58:59, :].tolist(), axis=0)[1]])
            facial5points.append([np.average(gt_landmarks_array[73:74, :].tolist(), axis=0)[0],
                                  np.average(gt_landmarks_array[73:74, :].tolist(), axis=0)[1]])
            # 获取crop后的图片和对应仿射矩阵。
            warped_face, tfm = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            if not os.path.isdir(os.path.join(target_root, subfolder)):
                os.mkdir(os.path.join(target_root, subfolder))
            img_warped.save(os.path.join(target_root, subfolder, lbl.split('.')[0] + '.png'))
            
        # 对图片进行旋转镜像，创建新的lbl文件夹。
        #     degrees = ['r','90','90_r','180','180_r','270','270_r']
            degrees = ['r']
            for degree in degrees:
                new_image_name = image_name+'_'+degree
                print("Processing\t{}".format(os.path.join(target_root, subfolder, new_image_name)))
                if not os.path.isdir(os.path.join(target_root,new_image_name)):
                    os.mkdir(os.path.join(target_root,new_image_name))
                # 获取旋转角度
                if degree == 'r':
                    angle = 0
                else:
                    angle = int(degree.split('_')[0])
                img_new = img_warped.rotate(angle)
                if 'r' in degree:
                    img_new = img_new.transpose(Image.FLIP_LEFT_RIGHT)
                img_new.save(os.path.join(target_root,new_image_name,new_image_name+'_'+lbl.split('_')[-1] ))