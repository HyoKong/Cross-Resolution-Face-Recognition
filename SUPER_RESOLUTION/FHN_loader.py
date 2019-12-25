import torch
from torch.utils import data
from torchvision import transforms

import os
import math
import random
import numpy as np

from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

import pandas as pd

class CelebA_HQ_loader(data.Dataset):
    '''
    Dataloader for CelebA-HQ dataset.
    '''

    def __init__(self,
                 parsing_root=r'',
                 landmark_root='',
                 img_root=r'D:\Hyo_Dataset\Cross-Resolution\CelebAMask-HQ\CelebA-HQ-img',
                 is_transform=True):
        self.img_root = img_root
        self.landmark_root = landmark_root
        self.parsing_root = parsing_root
        self.is_transform = is_transform
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.img_list = os.listdir(img_root)
        self.scale_list = [2, 4, 4, 8, 8, 8, 8, 8, 8]
        self.n_classes = 13

        # self.df = pd.read_csv(self.landmark_root)

    def __len__(self):
        return len(os.listdir(self.img_root))

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.img_root, img_name)

        random_angle = random.uniform(-180, 180)
        random_contrast = random.uniform(0.93, 1.07)
        random_brightness = random.uniform(0.92, 1.08)
        random_sharpness = random.uniform(0.93, 1.07)

        # ****************  SR Image    ***************
        sr_img = Image.open(img_path).resize((128, 128))

        # random rotate and crop
        sr_img = sr_img.rotate(random_angle)
        nh = random.randint(0, 128 - 112)
        nw = random.randint(0, 128 - 112)
        sr_img = sr_img.crop((nw, nh, nw + 112, nh + 112))

        # ****************  LR Image    ****************
        scale = random.choice(self.scale_list)
        lr_img = sr_img.resize((int(128/scale), int(128/scale))).resize((112, 112), Image.ANTIALIAS)

        # ****************  Augmentation    *****************
        contrast = ImageEnhance.Contrast(sr_img)
        sr_img = contrast.enhance(random_contrast)

        brightness = ImageEnhance.Contrast(sr_img)
        sr_img = brightness.enhance(random_brightness)

        sharpness = ImageEnhance.Contrast(sr_img)
        sr_img = sharpness.enhance(random_sharpness)

        contrast = ImageEnhance.Contrast(lr_img)
        lr_img = contrast.enhance(random_contrast)

        brightness = ImageEnhance.Contrast(lr_img)
        lr_img = brightness.enhance(random_brightness)

        sharpness = ImageEnhance.Contrast(lr_img)
        lr_img = sharpness.enhance(random_sharpness)

        # ****************  Parsing Map     **********************
        parsing_label = Image.open(self.parsing_root, img_name.split('.')[0].zfill(5) + '.png').resize((128, 128)).rotate(
            (random_angle)).crop((nw, nh, nw + 112, nh + 112))

        # *****************     Landmark    *********************
        landmark = self.df.loc[img_name].values[0].reshape(-1, 2)
        landmark = np.dot(landmark - 64, self.rotate_matrix(random_angle)) + 64
        landmark[:, 0] = landmark[:, 0] - int(nw)
        landmark[:, 1] = landmark[:, 1] - int(nh)
        heatmap = self.generate_hm(height=56, width=56, landmark=landmark, s=2.0)

        if self.is_transform:
            lr_img, sr_img, parsing_label, heatmap = self.transform(lr_img, sr_img, parsing_label, heatmap)

        batch = {}
        batch['lr_img'] = lr_img
        batch['hr_img'] = sr_img
        batch['parsing_label'] = parsing_label
        batch['heatmap'] = heatmap

        return batch

    def rotate_matrix(self, angle):
        # get affine matrix
        angle = angle / 180.0 * math.pi
        matrix = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        return matrix

    def generate_hm(self, height, width, landmark, s=2.0):
        # Generate a full heatmap for every landmarks in an array.
        N = np.shape(landmark)[0]
        # hm = np.zeros((N,height,width),dtype=np.float32)
        # for i in range(N):
        #     hm[i,:,:] = self.gaussian_k(landmark[i][0],landmark[i][1],s,height,width)
        hm = np.zeros((height, width), dtype=np.float32)
        for i in range(N):
            hm += self.gaussian_k(landmark[i][0], landmark[i][1], sigma=s, height=height, width=width)
            # hm += self.gaussian_k((landmark[i,0], landmark[i,1], s, height, width))
        return hm

    def gaussian_k(self, x0, y0, sigma, width=224, height=224):
        """
        Make a suqare gaussian kernel centered at (x0,y0) with sigma.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def transform(self, lr_img, sr_img, lbl, landmark):
        lr_img = self.tf(lr_img)
        sr_img = self.tf(sr_img)
        lbl = torch.from_numpy(np.expand_dims(np.array(lbl), axis=0)).long()
        landmark = torch.from_numpy(landmark.astype(np.float32))
        lbl[lbl == 255] = 0
        return lr_img, sr_img, lbl, landmark

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (11, 3)
        """
        return np.asarray(
            [
                [255, 255, 255],
                [255, 182, 147],
                [128, 128, 192],
                [192, 0, 0],
                [0, 0, 128],
                [0, 142, 64],
                [255, 255, 0],
                [128, 0, 128],
                [255, 128, 255],
                [255, 0, 0],
                [79, 39, 0],
                [0, 255, 255],
                [255, 0, 255],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask1 = mask / (mask.max() - mask.min()) * 11.0
        # pdb.set_trace()
        if mask1.min() < 0:
            mask1 += mask1.min()
        mask = mask1.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == ii, axis=-1))[:2]] = label
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False, save=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()

        label_mask1 = label_mask.astype(int)

        r = label_mask1.copy()
        g = label_mask1.copy()
        b = label_mask1.copy()
        for ll in range(0, self.n_classes):
            r[label_mask1 == ll] = label_colours[ll, 0]
            g[label_mask1 == ll] = label_colours[ll, 1]
            b[label_mask1 == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask1.shape[0], label_mask1.shape[1], 3))
        rgb[:, :, 0] = r[:, :, 0] / 255.0
        rgb[:, :, 1] = g[:, :, 1] / 255.0
        rgb[:, :, 2] = b[:, :, 2] / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

class Helen_Loader(data.Dataset):
    '''
    Dataloader for Helen dataset.
    '''

if __name__ == '__main__':
    h = CelebA_HQ_loader()
    parsing_label = Image.open(r'D:\Hyo_Dataset\Cross-Resolution\CelebAMask-HQ\CelebAMask-HQ-mask-anno\segmentation\00017.png')
    labelmask = np.array(parsing_label)
    h.decode_segmap(labelmask, plot=True)