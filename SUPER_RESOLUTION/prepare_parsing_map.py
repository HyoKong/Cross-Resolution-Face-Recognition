import os
import shutil
import numpy as np
from PIL import Image

parsing_root = r'D:\Hyo_Dataset\Cross-Resolution\CelebAMask-HQ\CelebAMask-HQ-mask-anno\total'
segmentation_root = r'D:\Hyo_Dataset\Cross-Resolution\CelebAMask-HQ\CelebAMask-HQ-mask-anno\segmentation'

combination_dict = {}  # {id1:[root1, root2,...],id2:[]}
id_list = []

for img in os.listdir(parsing_root):
    img_path = os.path.join(parsing_root, img)
    id = img.split('_')[0]
    if id not in id_list:
        id_list.append(id)

label_dict = {}
label_dict['_hair.png'] = 1
label_dict['_skin.png'] = 2
label_dict['_u_lip.png'] = 3
label_dict['_l_lip.png'] = 4
label_dict['_l_eye.png'] = 5
label_dict['_r_eye.png'] = 6
label_dict['_l_ear.png'] = 7
label_dict['_r_ear.png'] = 8
label_dict['_nose.png'] = 9
label_dict['_mouth.png'] = 10
label_dict['_r_brown.png'] = 11
label_dict['_l_brown.png'] = 12

for id in id_list:
    new_array = np.zeros([512, 512, 3], dtype='uint8')
    for k, v in label_dict.items():
        img_path = os.path.join(parsing_root, id + k)
        if os.path.exists(img_path):
            label_img = Image.open(img_path)
            img_array = np.array(label_img)
            new_array = np.where(img_array < 255, new_array, v)
    img = Image.fromarray(new_array, 'RGB')
    print(id)
    img.save(os.path.join(segmentation_root, id + '.png'))

'''
# copy to one dictionary
target = r'D:\Hyo_Dataset\Cross-Resolution\CelebAMask-HQ\CelebAMask-HQ-mask-anno\total'
for dir in os.listdir(parsing_root):
    dir_root = os.path.join(parsing_root, dir)
    for img in os.listdir(dir_root):
        img_path = os.path.join(dir_root,img)
        try:
            shutil.copy(img_path,os.path.join(target,img))
        except:
            print(img_path)
'''

# for dir in os.listdir(parsing_root):
#     dir_root = os.path.join(parsing_root, dir)

# print(0)
