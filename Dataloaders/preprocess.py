import cv2
import math
import torch
import random

import numpy as np


class CropResize(object):
    """
        先按照一定的比例进行裁剪, 然后再缩放到目标尺寸, 图像一般使用线性插值, 
    分割掩膜一般使用最近邻插值。
    """
    def __init__(self, crop_rate, dst_size):
        self.crop_rate = crop_rate
        self.dst_size = dst_size

    def __call__(self, data):
        if isinstance(data, list):
            result_list = []
            for single_data in data:
                assert single_data["image"].shape == single_data["label"].shape    
                single_data["image"] = self._operation(single_data["image"], interpolation=cv2.INTER_LINEAR)
                single_data["label"] = self._operation(single_data["label"], interpolation=cv2.INTER_NEAREST)
                result_list.append(single_data)
            return result_list
        
        elif isinstance(data, dict):
            assert data["image"].shape == data["label"].shape
            data["image"] = self._operation(data["image"], interpolation=cv2.INTER_LINEAR)
            data["label"] = self._operation(data["label"], interpolation=cv2.INTER_NEAREST)
            return data

        else:
            self._operation(data)
            return data

    def _operation(self, data, interpolation=cv2.INTER_NEAREST):
        h, w = data.shape
        ty = round(self.crop_rate[0]*h)
        by = round(self.crop_rate[1]*h)
        lx = round(self.crop_rate[2]*w)
        rx = round(self.crop_rate[3]*w)
        data = data[ty:by, lx:rx]
        data = cv2.resize(data, self.dst_size, interpolation=interpolation)
        return data


class WindowCenterWidth(object):
    """
        窗宽窗位, width: 医学图像像素值宽; center: 医学图像像素值位, 将 uint16 格式的
    医学图像数据转化为 uint8 格式的常规图像数据。
    """
    def __init__(self, center, width):
        self.center = center
        self.width = width
    
    def __call__(self, data):
        if isinstance(data, list):
            assert isinstance(self.center, list) and isinstance(self.width, list)
            result_list = []
            for single_data, center, width in zip(data, self.center, self.width):
                single_data["image"] = self._operation(single_data["image"], center, width)
                result_list.append(single_data)
            return result_list

        elif isinstance(data, dict):
            data["image"] = self._operation(data["image"], self.center, self.width)
            return data
        
        else:
            data = self._operation(data, self.center, self.width)
            return data
            
    def _operation(self, data, center, width):
        min_pixel = center - width/2
        max_pixel = center + width/2

        higher_mask = data > max_pixel
        lower_mask = data < min_pixel
        
        data = data.astype(np.float32)
        data = (data - min_pixel)*255 / (max_pixel - min_pixel)
        data = np.around(data, decimals=0)
        data = data.astype(np.uint8)
        data[lower_mask] = 0
        data[higher_mask] = 255

        return data
            

class MagnitudeLimitFilter(object):
    """
        限幅滤波器
    """
    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, data):
        if isinstance(data, list):
            result_list = []
            for single_data in data:
                single_data["image"] = self._operation(single_data["image"])
                result_list.append(single_data)
            return result_list

        elif isinstance(data, dict):
            data["image"] = self._operation(data["image"])
            return data
        
        else:
            data = self._operation(data)
            return data

    def _operation(self, data):
        data[data <= self.thresh] = 0
        return data


class RandomRotate(object):
    def __init__(self, rotate_angel):
        self.rotate_angel = rotate_angel

    def __call__(self, data):
        angel = random.randint(0, self.rotate_angel+1)

        if isinstance(data, list):
            result_list = []
            for single_data in data:
                single_data["image"] = self._operation(single_data["image"], angel, interpolation=cv2.INTER_LINEAR)
                single_data["label"] = self._operation(single_data["label"], angel, interpolation=cv2.INTER_NEAREST)
                result_list.append(single_data)
            
            return result_list
        elif isinstance(data, dict):
            data["image"] = self._operation(data["image"], angel, interpolation=cv2.INTER_LINEAR)
            data["label"] = self._operation(data["label"], angel, interpolation=cv2.INTER_NEAREST)

            return data
        else:
            data = self._operation(data, angel)

            return data

    def _operation(self, data, angel, interpolation=cv2.INTER_NEAREST):
        h, w = data.shape[0], data.shape[1]
        rotate_matrix = cv2.getRotationMatrix2D((h//2, w//2), angel, 1)
        data = cv2.warpAffine(data, rotate_matrix, (h, w), flags=interpolation)

        return data


class RandomFlip(object):
    def __init__(self, flag, rate):
        self.flag = flag
        self.rate = rate

    def __call__(self, data):
        # assert rate*10 == math.floor(rate*10), "One decimal place of flip rate after the decimal point."
        sample = random.random()
        if sample < rate:
            if self.flag == 1:
                flip_type = 1
            elif self.flag == 2:
                flip_type = 0
            elif self.flag == 3:
                flip_type = -1
            elif self.flag == 4:
                flip_type = random.randint(-1, 1)

        if isinstance(data, list):
            result_list = []
            for single_data in data:
                single_data["image"] = self._operation(single_data["image"], flip_type)
                single_data["label"] = self._operation(single_data["label"], flip_type)
                result_list.append(single_data)
            
            return result_list
        elif isinstance(data, dict):
            data["image"] = self._operation(data["image"], flip_type)
            data["label"] = self._operation(data["label"], flip_type)

            return data
        else:
            data = self._operation(data, flip_type)

            return data

    def _operation(self, data, flip_type):
        data = cv2.flip(data, flip_type)

        return data


class RandomTranslate(object):
    def __init__(self, translate_range):
        self.translate_range = translate_range

    def __call__(self, data):
        x_translate_dis = random.randint(-1*self.translate_range, self.translate_range)
        y_translate_dis = random.randint(-1*self.translate_range, self.translate_range)

        if isinstance(data, list):
            result_list = []
            for single_data in data:
                single_data["image"] = self._operation(single_data["image"], x_translate_dis, y_translate_dis, interpolation=cv2.INTER_LINEAR)
                single_data["label"] = self._operation(single_data["label"], x_translate_dis, y_translate_dis, interpolation=cv2.INTER_NEAREST)
                result_list.append(single_data)
            
            return result_list
        elif isinstance(data, dict):
            data["image"] = self._operation(data["image"], x_translate_dis, y_translate_dis, interpolation=cv2.INTER_LINEAR)
            data["label"] = self._operation(data["label"], x_translate_dis, y_translate_dis, interpolation=cv2.INTER_NEAREST)

            return data
        else:
            data = self._operation(data, x_translate_dis, y_translate_dis)

            return data

    def _operation(self, data, x_translate_dis, y_translate_dis, interpolation=cv2.INTER_NEAREST):
        h, w = data.shape[0], data.shape[1]
        translate_matrix = np.float32([[1, 0, x_translate_dis], [0, 1, y_translate_dis]])
        data = cv2.warpAffine(data, translate_matrix, (h, w), flags=interpolation)

        return data


class Normalize(object):
    def __call__(self, data):
        if isinstance(data, list):
            result_list = []
            for single_data in data:
                single_data["image"] = self._operation(single_data["image"])
                result_list.append(single_data)
            return result_list

        elif isinstance(data, dict):
            data["image"] = self._operation(data["image"])
            return data
        
        else:
            data = self._operation(data)
            return data

    def _operation(self, data):
        data = data/255
        return data


class ToTensor(object):
    def __init__(self, if_cuda):
        self.if_cuda = if_cuda

    def __call__(self, data):
        if isinstance(data, list):
            result_list = []
            for single_data in data:
                single_data["image"] = torch.from_numpy(single_data["image"]).float().unsqueeze(dim=0)
                single_data["label"] = torch.from_numpy(single_data["label"].astype(np.uint8))
                if self.if_cuda:
                    single_data["image"] = single_data["image"].cuda()
                    single_data["label"] = single_data["label"].cuda()
                result_list.append(single_data)
            return result_list

        elif isinstance(data, dict):
            data["image"] = torch.from_numpy(data["image"]).float().unsqueeze(dim=0)
            data["label"] = torch.from_numpy(data["label"].astype(np.uint8))
            if self.if_cuda:
                data["image"] = data["image"].cuda()
                data["label"] = data["label"].cuda()
            return data