import os
import cv2
import nrrd
import math
import shutil

import numpy as np
import SimpleITK as sitk


normal_organ_name_list = ['Anal Canal', 'Bladder', 'Rectum', 'Femoral Head(L)', 'Femoral Head(R)']


def str_part_in(sub_str, str_list, case_omit=True):
    '''
        对于 element in list 这样的语句, 需要 element 与 list 中的某一个 item 完全相同才输出为 True, 
    而 part_in , 即 sub_str in list 则只需要 sub_str 与 list 中的某一个 item 满足 sub_str in item 即可。
    输入:
        str_list: 包含有多个 item 的 list。
        sub_str: 需要进行 part_in 判断的字符串。
        case_omit: 是否忽略大小写。
    输出:
        flag: 是否满足 str_list 与 sub_str 之间 part_in 的条件。
    '''
    flag = False
    for str_ in str_list:    
        if str_in(sub_str, str_, case_omit):
            flag = True

    return flag


def str_in(sub_str, main_str, case_omit=True):
    '''
        对于只想要让 sub_str 中的所有字母按顺序在 main_str 中出现就算满足 sub_str in main_str 的情况, 
    需要一个 str_in 函数来实现。
    输入:
        main_str: 主字符串。
        sub_str: 需要判断是否满足 str_in 的字符串。
        case_omit: 是否忽略大小写。
    输出:
        flag: 是否满足 main_str 和 sub_str 之间的 str_in 的条件。
    '''
    if case_omit:
        main_str = main_str.upper()
        sub_str = sub_str.upper()
    
    flag = False
    main_index = 0; sub_index = 0;
    while(main_index != len(main_str)):
        if sub_index == len(sub_str): break
        if main_str[main_index] == sub_str[sub_index]:
            sub_index += 1
        main_index += 1

    if sub_index == len(sub_str): flag = True

    return flag


def Context_Max_Similar_Calculation(sub_str, main_str, case_omit=True):
    if case_omit:
        main_str = main_str.upper()
        sub_str = sub_str.upper()

    main_index = 0; sub_index = 0;
    while(main_index != len(main_str)):
        if sub_index == len(sub_str): break
        if main_str[main_index] == sub_str[sub_index]:
            sub_index += 1
        main_index += 1

    match_rate = sub_index/len(sub_str)

    return match_rate


def Context_Max_Similar_Selection(main_str, select_str_list, case_omit=True, conf_thresh=0.7):
    max_match_rate = -1
    max_match_index = -1
    for index in range(len(select_str_list)):
        sub_str = select_str_list[index]
        new_match_rate = Context_Max_Similar_Calculation(sub_str, main_str, case_omit)
        if new_match_rate > max_match_rate:
            max_match_rate = new_match_rate
            max_match_index = index

    if max_match_rate > conf_thresh:
        return select_str_list[max_match_index]
    else:
        return None


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


def convert_NiFTI_to_NRRD(NiFTI_path, NRRD_path):
    """
        转换为 NRRD 时的参考: http://teem.sourceforge.net/nrrd/format.html#space
    """
    _, nrrd_options = nrrd.read(f"{os.path.dirname(__file__)}/template.seg.nrrd")
    array, origin, spacing = read_NiFTI(NiFTI_path)
    raw_n, raw_r, raw_c = array.shape

    z_bottom, z_top = search_reverse(array, raw_n, 0, 0)
    y_front, y_behind = search_reverse(array, raw_r, 0, 1)
    x_left, x_right = search_reverse(array, raw_c, 0, 2)

    print("offset: z range: {:>3}, {:>3} y range: {:>3}, {:>3} x range: {:>3}, {:>3}".format(z_bottom, z_top, y_front, y_behind, x_left, x_right))

    array = array[z_bottom:(z_top+1), y_front:(y_behind+1), x_left:(x_right+1)].transpose(2, 1, 0)
    data = []

    c, r, n = array.shape

    for i in range(len(normal_organ_name_list)):
        value_mask = (array == i+1).astype(np.uint8)
        x_range = search(value_mask, c, 1, 0)
        y_range = search(value_mask, r, 1, 1)
        z_range = search(value_mask, n, 1, 2)
        nrrd_options[f"Segment{i}_Extent"] = "{} {} {} {} {} {}".format(*x_range, *y_range, *z_range)
        nrrd_options[f"Segment{i}_Name"] = normal_organ_name_list[i]
        data.append(np.expand_dims(value_mask, axis=0))
        print("organ {:>20} range: x range {:>3}, {:>3} y range {:>3}, {:>3} z range {:>3}, {:>3}".format(normal_organ_name_list[i], *x_range, *y_range, *z_range))
    
    print()
    data = np.concatenate(data, axis=0)

    nrrd_options["sizes"] = np.array(data.shape)

    nrrd_options["space directions"][1][0] = spacing[0]
    nrrd_options["space directions"][2][1] = spacing[1]
    nrrd_options["space directions"][3][2] = spacing[2]

    nrrd_options["Segmentation_ReferenceImageExtentOffset"] = "{} {} {}".format(x_left, y_front, z_bottom)

    nrrd_options["space"] = "left-posterior-superior"
    # nrrd_options["space"] = "right-anterior-superior"

    origin_fixed = [origin[0]+x_left*spacing[0], origin[1]+y_front*spacing[1], origin[2]+z_bottom*spacing[2]]
    # origin_fixed = [-1*origin[0]-x_left*spacing[0], -1*origin[1]-y_front*spacing[1], origin[2]+z_bottom*spacing[2]]

    nrrd_options["space origin"] = np.array(origin_fixed)

    nrrd.write(NRRD_path, data, nrrd_options)


def list_avg(item_list):
    return sum(item_list)/len(item_list)


def list_stdev(item_list):
    avg = list_avg(item_list)
    item_list = [(x - avg)**2 for x in item_list]

    return math.sqrt(sum(item_list) / len(item_list))