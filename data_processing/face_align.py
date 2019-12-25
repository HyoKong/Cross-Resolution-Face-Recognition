from PIL import Image
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "face alignment")
    parser.add_argument("-source_root", "--source_root", help = "specify your source dir", default = "/media/hyo/文档/VIS-NIR/SKETCH/CUFS/CUHK_training_sketch/sketch", type = str)
    parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default = "/media/hyo/文档/VIS-NIR/SKETCH/CUFS/CUHK_training_sketch/sketch_aligned", type = str)
    parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default = 112, type = int)
    args = parser.parse_args()

    source_root = args.source_root # specify your source dir
    dest_root = args.dest_root # specify your destination dir
    crop_size = args.crop_size # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale

    # cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    # os.chdir(source_root)
    # os.system("find . -name '*.DS_Store' -type f -delete")
    # os.chdir(cwd)
    
    # source_root = 's3_NIR_20385_008.bmp'
    # source_root = 's1_VIS_00003_010.jpg'
    # target_root = 's3_NIR_20385_008.jpg'
    # img = Image.open(source_root)
    # img.save(target_root,'jpeg',quality=100)
    # img = Image.open(target_root)
    # img = img.convert('RGB')
    # arr_img = np.array(img)
    # # img = np.expand_dims(img, 0)
    # _, landmarks = detect_faces(img)
    # facial5points = [[landmarks[0][j], landmarks[0][j+5]] for j in range(5)]
    # warped_face = warp_and_crop_face(np.array(img),facial5points,reference,crop_size=(crop_size,crop_size))
    # img_warped = Image.fromarray(warped_face[0])
    # plt.imshow(img_warped)
    # plt.show()

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)


    for image_name in os.listdir(source_root):
        print("Processing\t{}".format(os.path.join(source_root, image_name)))
        img = Image.open(os.path.join(source_root, image_name))
        
        if not os.path.isdir(os.path.join(dest_root,image_name.split('.')[0])):
            os.mkdir(os.path.join(dest_root,image_name.split('.')[0]))
            
        img.save(os.path.join(dest_root,image_name.split('.')[0],image_name.split('.')[0]+'.jpg'),'jpeg',quality=100)
        
        img = Image.open(os.path.join(dest_root,image_name.split('.')[0],image_name.split('.')[0]+'.jpg'))
        img = img.convert('RGB')
        arr_img = np.array(img)
        
        try: # Handle exception
            _, landmarks = detect_faces(img)
        except Exception:
            print("{} is discarded due to exception!".format(os.path.join(source_root, image_name)))
            continue
        if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
            print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, image_name)))
            continue
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
        img_warped = Image.fromarray(warped_face[0])
        if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
            image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
        img_warped.save(os.path.join(dest_root,image_name.split('.')[0],image_name.split('.')[0]+'.jpg'),quality=100)
        
        img = Image.open(os.path.join(dest_root,image_name.split('.')[0],image_name.split('.')[0]+'.jpg'))

        _, landmark = detect_faces(img)
        if len(landmark)==0:
            print('abcdefghijklmnopqrstuvwxyz')
            continue
        else:
            facial_5_points = [[landmark[0][j], landmark[0][j + 5]] for j in range(5)]
            
            left_eye_landmark = facial_5_points[0]
            left_eye_img = img.crop((left_eye_landmark[0] - 11, left_eye_landmark[1] - 11, left_eye_landmark[0] + 11,
                            left_eye_landmark[1] + 11))
            left_eye_img.save(os.path.join(dest_root,image_name.split('.')[0],'left_eye.jpg'),quality=100)
            print(image_name,'  left eye')
    
            right_eye_landmark = facial_5_points[1]
            right_eye_img = img.crop((right_eye_landmark[0] - 11, right_eye_landmark[1] - 11, right_eye_landmark[0] + 11,
                                      right_eye_landmark[1] + 11))
            right_eye_img.save(os.path.join(dest_root, image_name.split('.')[0], 'right_eye.jpg'), quality=100)
            print(image_name, '  right eye')
