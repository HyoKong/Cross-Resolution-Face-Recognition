import os
import math
import random
import numpy as np
import face_alignment
from skimage import io
from skimage.transform import resize
import collections
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

'''
# get affine matrix
def rotate_matrix(angle):
    angle = angle / 180.0 * math.pi
    matrix = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    return matrix


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

plot_style = dict(marker='o', markersize=4, linestyle='-', lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])

pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),

              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),

              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),

              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),

              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),

              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),

              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),

              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),

              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))

              }

# img = io.imread(r'1.jpg')
img = Image.open(r'1.jpg').resize((128, 128))

# img = resize(img, (112, 112))
preds = fa.get_landmarks(np.array(img))[-1]
# #
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 1, 1)

# ***********************   rotate  ***********************
random_angle = random.uniform(-20, 20)
# img = Image.fromarray(img, 'RGB').resize((112, 112))
img = img.rotate(random_angle)
preds = np.dot(preds - 64, rotate_matrix(random_angle)) + 64

# ***********************   crop    **************************
nh = random.randint(0, 128 - 112)
nw = random.randint(0, 128 - 112)
img = img.crop((nw, nh, nw + 112, nh + 112))
preds[:,0] = preds[:,0] - int(nw)
preds[:,1] = preds[:,1] - int(nh)

ax.imshow(img)
for pred_type in pred_types.values():
    ax.plot(preds[pred_type.slice, 0],

            preds[pred_type.slice, 1],

            color=pred_type.color, **plot_style)

ax.axis('off')
plt.savefig('1_new.png',dpi=100,pad_inches = 0,bbox_inches='tight')

plt.show()
'''
columns = []
for i in range(136):
    columns.append(str(i))

df = pd.DataFrame(columns=columns)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0')

img_root = r'D:\Hyo_Dataset\Cross-Resolution\CelebAMask-HQ\CelebA-HQ-img'
for img_name in os.listdir(img_root):
    img_path = os.path.join(img_root, img_name)
    img = Image.open(img_path).resize((128,128))
    preds = fa.get_landmarks(np.array(img))[-1]
    preds = preds.reshape(-1)
    df.loc[img_name] = preds
    print(img_name)
df.to_csv('landmark.csv',header=True,index=True)
print(0)
