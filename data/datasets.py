import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

class FIREDataset(Dataset):
    def __init__(self, data_path, size, transforms):
        self.data_path = data_path
        self.transforms = transforms

        self.data_num_list = get_data_num_by_split('train', filter='AS')
        self.img1, self.img2, self.theta, self.control_points = [], [], [], []
        self.size = size
        print('%2d image pairs have been loaded for training' % len(self.data_num_list))
    
    def __getitem__(self, index):
        dn = self.data_num_list[index]
        img1 = np.array(Image.open(self.data_path + '/Gray_256/' + dn + '_1.jpg')).astype(np.float32)
        img2 = np.array(Image.open(self.data_path + '/Gray_256/' + dn + '_2.jpg')).astype(np.float32)
        # img1 = img1 / 255
        # img2 = img2 / 255
        
        x = np.ascontiguousarray(img1[None, ...])
        y = np.ascontiguousarray(img2[None, ...])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y
    
    def __len__(self):
        return len(self.data_num_list)

    def preprocess_matrix(self, tf):
        tf[0][2] = tf[0][2] * self.size / 2912
        tf[1][2] = tf[1][2] * self.size / 2912
        b = np.array([0, 0, 1])
        M = np.r_[tf, [b]]
        T = np.array([[2 / self.size, 0, -1],
                      [0, 2 / self.size, -1],
                      [0, 0, 1]])
        theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
        theta = torch.Tensor(theta[:2, :])
        return theta


class FIREInferDataset(Dataset):
    def __init__(self, data_path, size, transforms):
        self.data_path = data_path
        self.transforms = transforms

        self.data_num_list = get_data_num_by_split('test', filter='AS')
        self.img1, self.img2, self.theta, self.control_points = [], [], [], []
        self.size = size
        print('%2d image pairs have been loaded for testing' % len(self.data_num_list))
    
    def __getitem__(self, index):
        dn = self.data_num_list[index]
        img1 = np.array(Image.open(self.data_path + '/Gray_256/' + dn + '_1.jpg')).astype(np.float32)
        img2 = np.array(Image.open(self.data_path + '/Gray_256/' + dn + '_2.jpg')).astype(np.float32)
        # img1 = img1 / 256
        # img2 = img2 / 256
        cimg1 = np.array(Image.open(self.data_path + '/color_256/' + dn + '_1.jpg')).astype(np.float32)
        cimg2 = np.array(Image.open(self.data_path + '/color_256/' + dn + '_2.jpg')).astype(np.float32)
        # cimg1 = cimg1 / 256 * 2 - 1
        # cimg2 = cimg2 / 256 * 2 - 1

        points = np.loadtxt(self.data_path + "/Ground Truth/control_points_" + dn + "_1_2.txt")
        control_points = torch.Tensor(points) * self.size / 2912
        
        x_gray = np.ascontiguousarray(img1[None, ...])
        y_gray = np.ascontiguousarray(img2[None, ...])
        x = np.ascontiguousarray(cimg1 / 256)
        y = np.ascontiguousarray(cimg2 / 256)
        x_gray, y_gray = torch.from_numpy(x_gray), torch.from_numpy(y_gray)
        # x, y = torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y).permute(2, 0, 1)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(x.shape, y.shape, x_gray.shape, y_gray.shape)
        return x, y, x_gray, y_gray, control_points
    
    def __len__(self):
        return len(self.data_num_list)

    def preprocess_matrix(self, tf):
        tf[0][2] = tf[0][2] * self.size / 2912
        tf[1][2] = tf[1][2] * self.size / 2912
        b = np.array([0, 0, 1])
        M = np.r_[tf, [b]]
        T = np.array([[2 / self.size, 0, -1],
                      [0, 2 / self.size, -1],
                      [0, 0, 1]])
        theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
        theta = torch.Tensor(theta[:2, :])
        return theta


def get_data_num():
    file_list = glob.glob('/data/student/nieqiushi/TransMorph_Transformer_for_Medical_Image_Registration/FIRE/Ground Truth/*')
    for i in range(len(file_list)):
        file_list[i] = file_list[i][-11:-8]
    return sorted(file_list)


def get_data_num_by_split(split, filter):
    data_num_list = []
    if split == 'all':
        data_num_list = get_data_num()
    elif split == 'train':
        data_num_list = ['A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'P01', 'P02', 'P03',
                'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P16', 'P17', 'P18', 'P20', 'P21',
                'P22', 'P23', 'P24', 'P25', 'P27', 'P28', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37',
                'P40', 'P41', 'P42', 'P43', 'P44', 'P45', 'P47', 'P48', 'P49', 'S01', 'S02', 'S03', 'S04', 'S05',
                'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19',
                'S20', 'S21', 'S22', 'S23', 'S25', 'S26', 'S27', 'S29', 'S30', 'S32', 'S34', 'S35', 'S36', 'S37',
                'S38', 'S39', 'S40', 'S42', 'S44', 'S46', 'S47', 'S48', 'S49', 'S50', 'S51', 'S52', 'S55', 'S57',
                'S58', 'S59', 'S60', 'S61', 'S62', 'S65', 'S67', 'S69', 'S71']
    elif split == 'test':
        data_num_list = ['A01', 'A02', 'A14', 'P04', 'P05', 'P15', 'P19', 'P26', 'P29', 'P38', 'P39', 'P46', 'S24', 'S28',
                'S31', 'S33', 'S41', 'S43', 'S45', 'S53', 'S54', 'S56', 'S63', 'S64', 'S66', 'S68', 'S70']
    else:
        raise Exception("wrong input for 'split'")
    result = []
    for data_num in data_num_list:
        if data_num[0] in filter:
            result.append(data_num)
    return result


class RaFDDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_gray, y_gray = pkload(path)
        x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        x_gray, y_gray = self.transforms([x_gray, y_gray])
        #plt.figure()
        #plt.imshow(x_gray[0], cmap='gray')
        #plt.show()
        x = np.ascontiguousarray(x_gray)
        y = np.ascontiguousarray(y_gray)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class RaFDInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_gray, y_gray = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        x_gray = np.ascontiguousarray(x_gray.astype(np.float32))
        y_gray = np.ascontiguousarray(y_gray.astype(np.float32))
        x = np.ascontiguousarray(x.astype(np.float32))
        y = np.ascontiguousarray(y.astype(np.float32))
        x_gray, y_gray = torch.from_numpy(x_gray), torch.from_numpy(y_gray)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, x_gray, y_gray

    def __len__(self):
        return len(self.paths)