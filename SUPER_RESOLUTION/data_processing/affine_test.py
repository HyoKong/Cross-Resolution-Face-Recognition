import cv2
import numpy as np

from align.detector import  detect_faces
from align.align_trans import get_reference_facial_points,warp_and_crop_face
import os
from PIL import Image,ImageDraw
import pdb
# from pylab import plot
import pandas as pd
import tqdm
import argparse
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='change landmark')
    parser.add_argument('--source_root',default = '/media/hyo/文档/Dataset/Helen/')
    parser.add_argument('--crop_size',default=224)
    parser.add_argument('--landmark_dir',default='/media/hyo/文档/Dataset/annotation/')
    # parser.add_argument('--')
    parser.add_argument('--num_of_landmark',default=194)
    parser.add_argument('--image_dir_224',type=str, default='/media/hyo/文档/Dataset/Helen_aligned_224_renew_v2/')
    parser.add_argument('--csv_file',type=str,default='/home/hyo/original_landmarks.csv')
    args = parser.parse_args()

    image_dir = args.source_root
    landmark_dir = args.landmark_dir
    crop_size = args.crop_size
    num_of_landmark = args.num_of_landmark
    image_dir_224 = args.image_dir_224
    csv_file = args.csv_file
    # scale = crop_size / 112.
    scale = crop_size / 112.
    
    original_5_points = get_reference_facial_points(default_square=True)
    original_5_points[:, 1] = original_5_points[:, 1]
    reference = (original_5_points) * scale

    cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    os.chdir(image_dir)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)
    
    def landmark2csv():
        landmark_txt_list = os.listdir(landmark_dir)
        df_column = []
        df_index = []
        # df_column.append('image_name')
        for i in range(num_of_landmark*2):
            if i%2==0:
                df_column.append('part_x_'+str(int(i/2)))
            else:
                df_column.append('part_y_'+str(int((i-1)/2)))
    
        landmark_df = pd.DataFrame(columns=df_column)
        
        for each_txt in landmark_txt_list:
            with open(os.path.join(landmark_dir, each_txt), 'r') as f:
                print(each_txt)
                raw_landmark_list = f.readlines()
                landmark_list = []
                for each in raw_landmark_list[1:]:
                    temp = each.strip('\n').split(' , ')
                    # for each in temp:
                    #     each = float(each)
                    temp1 = [float(i) for i in temp]
                    landmark_list.append(temp1)
                l = np.array(landmark_list)
                landmark_list = l.flatten().tolist()
                # landmark_list.insert(0,raw_landmark_list[0].strip('\n'))
                landmark_df.loc[raw_landmark_list[0].strip('\n'),:] = landmark_list
                    # landmark_df.append(pd.DataFrame(landmark_list,index=[raw_landmark_list[0].strip('\n')]))
                f.close()
        landmark_df.to_csv('/home/hyo/test.csv')
    # landmark2csv()
    
    # 获得旋转后图片的名称
    def rotate_img_name(img_name,angle,if_flip=False):
        temp = img_name.split('.')
        angle = str(angle)
        if not if_flip:
            new_img_name = temp[0]+'_'+angle+'.jpg'
        else:
            new_img_name = temp[0]+'_'+angle+'_r.jpg'
        return new_img_name
    
    # 获得旋转仿射矩阵，90，180，270度
    def rotate_matrix(angle):
        # 角度变弧度
        angle = angle/180*math.pi
        matrix = [[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]]
        return matrix
    
    # 获得镜像后的landmark
    def flip_landmark(landmark):
        crop_size = args.crop_size
        flip_landmark = []
        temp = -landmark[0]
        flip_landmark.append(temp)
        flip_landmark.append(landmark[1])
        return flip_landmark
        
        
    df_landmark = pd.read_csv(csv_file)
    for subfolder in os.listdir(image_dir):
        for image_name in os.listdir(os.path.join(image_dir,subfolder)):
            print("Processing\t{}".format(os.path.join(image_dir_224, subfolder, image_name)))
            img = Image.open(os.path.join(image_dir,subfolder,image_name))
            # _, landmarks = detect_faces(img)
            #
            # facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    
            # 获取当前图片的仿射矩阵tsm
            facial5points = []
            gt_landmarks = df_landmark[df_landmark['img_name'] == image_name].values.flatten()[1:]
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
            
            warped_face,tfm = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            
            # 获取当前图片的原始landmark，把其存入二维数组，最后放入dataframe中
            landmark = df_landmark[df_landmark['img_name']==image_name].values[0]
            landmark_list = np.array(landmark[1:]).reshape(num_of_landmark,2).tolist()
            
            landmark_new = []
            landmark_new_r = []
            landmark_new_90 = []
            landmark_new_90_r = []
            landmark_new_180 = []
            landmark_new_180_r = []
            landmark_new_270 = []
            landmark_new_270_r = []
            for each in landmark_list:
                each.append(1)
                result = np.dot(tfm,np.array(each))
                # landmark_new.append(result.tolist())
                result_r = flip_landmark(result)
                result_r[0] = result_r[0]+224
                for _ in result:
                    landmark_new.append(_)
                #     if _<=224 and _>=0:
                #         landmark_new.append(_)
                #     else:
                #         pdb.set_trace()
                #         if math.fabs(_-224)-math.fabs(_-0)>=0:
                #             _ = 0.0
                #         else:
                #             _ = 224.0
                #         landmark_new.append(_)
                        
                for _ in result_r:
                    landmark_new_r.append(_)
                    # if _<=224 and _>=0:
                    #     landmark_new_r.append(_)
                    # else:
                    #     if math.fabs(_-224)-math.fabs(_-0)>=0:
                    #         _ = 0.0
                    #     else:
                    #         _ = 224.0
                    #     landmark_new_r.append(_)
            '''
                result_90 = np.dot(rotate_matrix(90),result)
                result_90_r = flip_landmark(result_90)
                result_90[0] = -result_90[0]
                result_90[1] = 224-result_90[1]
                result_90_r[0] = 224-result_90_r[0]
                result_90_r[1] = 224- result_90_r[1]
                
                for _ in result_90:
                    if _ <= 224 and _ >= 0:
                        landmark_new_90.append(_)
                    else:
                        if math.fabs(_-224)-math.fabs(_-0)>=0:
                            _ = 0.0
                        else:
                            _ = 224.0
                        landmark_new_90.append(_)
    
                for _ in result_90_r:
                    if _ <= 224 and _ >= 0:
                        landmark_new_90_r.append(_)
                    else:
                        if math.fabs(_-224)-math.fabs(_-0)>=0:
                            _ = 0.0
                        else:
                            _ = 224.0
                        landmark_new_90_r.append(_)
    
                
                result_180 = np.dot(rotate_matrix(180),result)
                result_180_r = flip_landmark(result_180)
                result_180[0] = 224+result_180[0]
                result_180[1] = 224+result_180[1]
                result_180_r[0] = result_180_r[0]
                result_180_r[1] = 224 + result_180_r[1]
                for _ in result_180:
                    if _ <= 224 and _ >= 0:
                        landmark_new_180.append(_)
                    else:
                        if math.fabs(_-224)-math.fabs(_-0)>=0:
                            _ = 0.0
                        else:
                            _ = 224.0
                        landmark_new_180.append(_)
    
                for _ in result_180_r:
                    if _ <= 224 and _ >= 0:
                        landmark_new_180_r.append(_)
                    else:
                        if math.fabs(_-224)-math.fabs(_-0)>=0:
                            _ = 0.0
                        else:
                            _ = 224.0
                        landmark_new_180_r.append(_)
    
                    
                result_270 = np.dot(rotate_matrix(270),result)
                result_270_r = flip_landmark(result_270)
                result_270[0] = 224 - result_270[0]
                result_270[1] = -result_270[1]
                result_270_r[0] = -result_270_r[0]
                result_270_r[1] = -result_270_r[1]
                for _ in result_270:
                    if _ <= 224 and _ >= 0:
                        landmark_new_270.append(_)
                    else:
                        if math.fabs(_-224)-math.fabs(_-0)>=0:
                            _ = 0.0
                        else:
                            _ = 224.0
                        landmark_new_270.append(_)
    
                for _ in result_270_r:
                    if _ <= 224 and _ >= 0:
                        landmark_new_270_r.append(_)
                    else:
                        if math.fabs(_-224)-math.fabs(_-0)>=0:
                            _ = 0.0
                        else:
                            _ = 224.0
                        landmark_new_270_r.append(_)
            '''
            # pdb.set_trace()
            landmark_new.insert(0,image_name)
            df_landmark[df_landmark['img_name'] == image_name] =  landmark_new
            landmark_new_r.insert(0,image_name.split('.')[0]+'_r.jpg')
            df_landmark.loc[df_landmark.shape[0],:] = landmark_new_r
            
            img_224 = Image.open(os.path.join(image_dir_224,image_name))
            img_224_r = img_224.transpose(Image.FLIP_LEFT_RIGHT)
            img_224_r.save(os.path.join(image_dir_224,image_name.split('.')[0]+'_r.jpg'))
            print(image_name+'----'+str(image_name.split('.')[0])+'_r.jpg')
            
            '''
            # 旋转图片并保存
            # 90度及镜像
            img_224_90 = img_224.rotate(90)
            img_224_90_r = img_224_90.transpose(Image.FLIP_LEFT_RIGHT)
            img_224_90_name = rotate_img_name(image_name,90,False)
            img_224_90_r_name = rotate_img_name(image_name,90,True)
            print(img_224_90_name+'----'+img_224_90_r_name)
            # img_224_90.save(os.path.join(image_dir_224,subfolder,img_224_90_name))
            # img_224_90_r.save(os.path.join(image_dir_224,subfolder,img_224_90_r_name))
            landmark_new_90.insert(0,img_224_90_name)
            landmark_new_90_r.insert(0,img_224_90_r_name)
            df_landmark.loc[df_landmark.shape[0],:] = landmark_new_90
            df_landmark.loc[df_landmark.shape[0],:] = landmark_new_90_r
            print('Row of csv:  '+str(df_landmark.shape[0]))
            
            # 180度及镜像
            # img_224 = Image.open(os.path.join(image_dir_224,subfolder,image_name))
            img_224_180 = img_224.rotate(180)
            img_224_180_r = img_224_180.transpose(Image.FLIP_LEFT_RIGHT)
            img_224_180_name = rotate_img_name(image_name,180,False)
            img_224_180_r_name = rotate_img_name(image_name,180,True)
            print(img_224_180_name+'----'+img_224_180_r_name)
            # img_224_180.save(os.path.join(image_dir_224,subfolder,img_224_180_name))
            # img_224_180_r.save(os.path.join(image_dir_224,subfolder,img_224_180_r_name))
            landmark_new_180.insert(0,img_224_180_name)
            landmark_new_180_r.insert(0,img_224_180_r_name)
            df_landmark.loc[df_landmark.shape[0],:] = landmark_new_180
            df_landmark.loc[df_landmark.shape[0],:] = landmark_new_180_r
            print('Row of csv:  ' + str(df_landmark.shape[0]))
    
            # 270度及镜像
            # img_224 = Image.open(os.path.join(image_dir_224,subfolder,image_name))
            img_224_270 = img_224.rotate(270)
            img_224_270_r = img_224_270.transpose(Image.FLIP_LEFT_RIGHT)
            img_224_270_name = rotate_img_name(image_name,270,False)
            img_224_270_r_name = rotate_img_name(image_name,270,True)
            print(img_224_270_name+'----'+img_224_270_r_name)
            # img_224_270.save(os.path.join(image_dir_224,subfolder,img_224_270_name))
            # img_224_270_r.save(os.path.join(image_dir_224,subfolder,img_224_270_r_name))
            landmark_new_270.insert(0,img_224_270_name)
            landmark_new_270_r.insert(0,img_224_270_r_name)
            df_landmark.loc[df_landmark.shape[0],:] = landmark_new_270
            df_landmark.loc[df_landmark.shape[0],:] = landmark_new_270_r
            print('Row of csv:  ' + str(df_landmark.shape[0]))
            '''
    
            
    df_landmark.to_csv('/home/hyo/xixixi_v3.csv')
    

    #
    # img = Image.open(image_root)
    # _ , landmarks = detect_faces(img)
    #
    # reference = get_reference_facial_points(default_square = True) * 2
    #
    # facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    # warped_face , tfm = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(224, 224))
    #
    # landmark_new = []
    # for each in landmark_list:
    #     each.append(1)
    #     result = np.dot(tfm,np.array(each))
    #     landmark_new.append(result.tolist())
    # # draw = ImageDraw.Draw(warped_face)
    # temp = Image.fromarray(np.uint8(warped_face))
    # draw = ImageDraw.Draw(temp)
    # for p in landmark_new:
    #     draw.line((p[0]-1,p[1]-1,p[0]+1,p[1]+1))
    # # for p in landmark_new:
    # #     temp.line(p[0]-1,p[0]+1,p[1]-1,p[1]+1)
    # # pdb.set_trace()
    # # plot(np.transpose(landmark_new)[0],np.transpose(landmark_new)[1])
    # # temp.show()
    # print('hhhhhh')