from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import scipy.misc as m

img_path = '../s1_VIS_00003_003.jpg'
img = Image.open(img_path)
tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

# img = img.convert('YCbCr')

# _ , landmark = detect_faces(img)
# facial_5_points = [[landmark[0][j],landmark[0][j+5]] for j in range(5)]
# print(facial_5_points)
# left_eye_landmark = facial_5_points[0]
# img = img.crop((left_eye_landmark[0]-11,left_eye_landmark[1]-11,left_eye_landmark[0]+11,left_eye_landmark[1]+11))

# imgArr = np.array(img)
# imgYArr = imgArr.copy()

imgArr = tf(img)
imgYArr = imgArr.clone()
gauss = np.random.normal(0.5,0.5,(3,112,112))
gauss = torch.from_numpy(gauss).float()
imgYArr = imgYArr + 0.2 * gauss
# imgYArr[:,:,0] = imgArr[:,:,0]*0.257 + imgArr[:,:,1]*0.564 + imgArr[:,:,2]*0.098 + 16/256.0
# imgYArr[:,:,1] = imgArr[:,:,0]*(-0.148) - imgArr[:,:,1]*0.291 + imgArr[:,:,2]*0.439 + 0
# imgYArr[:,:,2] = imgArr[:,:,0]*0.439 - imgArr[:,:,1]*0.268 - imgArr[:,:,2]*0.071 + 128/256.0

# imgYArr[0,:,:] = imgArr[0,:,:]*0.257 + imgArr[1,:,:]*0.564 + imgArr[2,:,:]*0.098 + (16/255.0 - 0.5)*2.0
# imgYArr[1,:,:] = imgArr[0,:,:]*(-0.148) - imgArr[1,:,:]*0.291 + imgArr[2,:,:]*0.439 + 128/255.0
# imgYArr[2,:,:] = imgArr[0,:,:]*0.439 - imgArr[1,:,:]*0.268 - imgArr[2,:,:]*0.071 + 128/255.0

imgYArr = imgYArr.cpu().numpy()
# imgArr = ((imgArr.cpu().numpy())/2.0+0.5)*255.0
#
# imgArr = imgArr[0,:,:]
# imgOri = np.array(img)[:,:,0]
# imgY = Image.fromarray(imgArr)
# image = m.toimage(imgYArr[0,:,:],cmin=-1,cmax=1)
image = m.toimage(imgYArr,cmin=-1,cmax=1)
# plt.imshow(imgY)
# plt.show()
image.show()
