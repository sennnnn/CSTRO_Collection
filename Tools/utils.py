import os
import cv2
import nrrd
import math
import shutil

import numpy as np
import SimpleITK as sitk


normal_organ_name_list = ['Brain Stem', 'Eye(L)', 'Eye(R)', 'Lens(L)', 'Lens(R)', 'Optic Nerve(L)', 'Optic Nerve(R)', 'Optic Chiasma', 'Temporal Lobes(L)', 'Temporal Lobes(R)', 'Pituitary', 'Parotid Gland(L)', 'Parotid Gland(R)', \
    'Inner Ear(L)', 'Inner Ear(R)', 'Mid Ear(L)', 'Mid Ear(R)', 'Jaw Joint(L)', 'Jaw Joint(R)', 'Spinal Cord', 'Mandible(L)', 'Mandible(R)']


def convert_gray_to_rgb(gray_image):
    gray_image = np.expand_dims(gray_image, axis=-1)
    rgb_image = np.repeat(gray_image, 3, axis=-1)

    return rgb_image


def read_array_from_NiFTI_path(nii_path):
    image = sitk.ReadImage(nii_path)
    array = sitk.GetArrayFromImage(image)

    return array


def read_NiFTI(nii_path):
    image = sitk.ReadImage(nii_path)
    array = sitk.GetArrayFromImage(image)
    origin, spacing = image.GetOrigin(), image.GetSpacing()

    return array, origin, spacing


def get_NiFTI(array, origin, spacing, dst_path=None):
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)

    if dst_path != None:
        sitk.WriteImage(image, dst_path)

    return image


def write_NiFTI(array, origin, spacing, dst_path):
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, dst_path)
    
    return image


def blend_mask(image, mask, alpha=0.5, coord=(0, 0)):
    m_h, m_w, _ = mask.shape
    for y in range(m_h):
        for x in range(m_w):
            image[coord[0]+y, coord[1]+x, 0] = (1-alpha)*image[coord[0]+y, coord[1]+x, 0] + alpha*mask[y, x, 0]
            image[coord[0]+y, coord[1]+x, 1] = (1-alpha)*image[coord[0]+y, coord[1]+x, 1] + alpha*mask[y, x, 1]
            image[coord[0]+y, coord[1]+x, 2] = (1-alpha)*image[coord[0]+y, coord[1]+x, 2] + alpha*mask[y, x, 2]

    return image


def parse_list(list_path):
    fp = open(list_path, "r")
    lines = fp.readlines()
    ret_list = [line.strip() for line in lines]

    return ret_list


def get_ROI_volume_from_NiFTI(nii_path):
    """
        关于器官体积的计算, 主要使用的方法是通过计算单个体素的体积, 然后累加器官包括的体素个数来近似
    器官体积, 输入的 NiFTI 文件中应该包含 ROI 勾画掩膜数据。
    输入:
        nii_path: NiFTI 文件的路径。
    输出:
        volume: 形如 [ROI_1_Volume, ROI_2_Volume, ...] 的列表, 以 mm^3 为单位。
    """
    label_NiFTI = sitk.ReadImage(nii_path)
    label = sitk.GetArrayFromImage(label_NiFTI)
    
    pixel_spacing = label_NiFTI.GetSpacing()
    voxel_volume = pixel_spacing[0]*pixel_spacing[1]*pixel_spacing[2]

    ROI_count = np.max(label) + 1
    volume = []
    for ROI_index in range(1, ROI_count):
        volume.append(len(label[label == ROI_index])*voxel_volume)

    return volume


def pad_resize(data, crop_rate, raw_shape):
    if isinstance(data, list):
        result_list = []
        for single_data, single_raw_shape in zip(data, raw_shape):
            container = np.zeros(single_raw_shape, dtype=np.uint8)
            h, w = raw_shape
            ty = round(crop_rate[0]*h)
            by = round(crop_rate[1]*h)
            lx = round(crop_rate[2]*w)
            rx = round(crop_rate[3]*w)
            single_data = cv2.resize(single_data, (by-ty+1, rx-lx+1), interpolation=cv2.INTER_NEAREST)
            container[ty:by, lx:rx] = single_data
            result_list.append(container)
        return result_list
    else:
        container = np.zeros(raw_shape, dtype=np.uint8)
        h, w = raw_shape
        ty = round(crop_rate[0]*h)
        by = round(crop_rate[1]*h)
        lx = round(crop_rate[2]*w)
        rx = round(crop_rate[3]*w)
        data = cv2.resize(data, (rx-lx, by-ty), interpolation=cv2.INTER_NEAREST)
        container[ty:by, lx:rx] = data
        return container


def window_center_acquire(mri_sequence_list):
    wc_list = []
    for sequence in mri_sequence_list:
        wc_list.append(settings.WINDOW_CENTER[sequence])
    
    if len(wc_list) == 1:
        wc_list = wc_list[0]

    return wc_list


def window_width_acquire(mri_sequence_list):
    wd_list = []
    for sequence in mri_sequence_list:
        wd_list.append(settings.WINDOW_WIDTH[sequence])

    if len(wd_list) == 1:
        wd_list = wd_list[0]

    return wd_list


def sequence_string_acquire(mri_sequence_list):
    if len(mri_sequence_list) == 3:
        string = "{}-{}-{}".format(*tuple(mri_sequence_list))
    elif len(mri_sequence_list) == 2:
        string = "{}-{}".format(*tuple(mri_sequence_list))
    elif len(mri_sequence_list) == 1:
        string = "{}".format(mri_sequence_list[0])

    return string


def search(array, length, value, dimension):
    array_shape = array.shape
    temp = [range(array_shape[index]) if index != dimension else [] for index in range(len(array_shape))]

    s = array_shape[dimension]
    e = 0
    for i in range(length):
        temp[dimension] = [i]
        index_mask = np.ix_(*tuple(temp))
        if np.max(array[index_mask]) == value:
            s = i
            break

    for i in range(length-1, -1, -1):
        temp[dimension] = [i]
        index_mask = np.ix_(*tuple(temp))
        if np.max(array[index_mask]) == value:
            e = i
            break    

    return s, e


def search_reverse(array, length, value, dimension):
    array_shape = array.shape
    temp = [range(array_shape[index]) if index != dimension else [] for index in range(len(array_shape))]
    
    s = array_shape[dimension]
    e = 0
    for i in range(length):
        temp[dimension] = [i]
        index_mask = np.ix_(*tuple(temp))
        if np.max(array[index_mask]) != value:
            s = i
            break

    for i in range(length-1, -1, -1):
        temp[dimension] = [i]
        index_mask = np.ix_(*tuple(temp))
        if np.max(array[index_mask]) != value:
            e = i
            break  

    return s, e


def list_avg(item_list):
    return sum(item_list)/len(item_list)


def list_stdev(item_list):
    avg = list_avg(item_list)
    item_list = [(x - avg)**2 for x in item_list]

    return math.sqrt(sum(item_list) / len(item_list))