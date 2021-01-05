import math

import numpy as np
import SimpleITK as sitk

from scipy.ndimage import morphology


def one_hot(array, num_classes, on_value=1, off_value=0):
    assert np.max(array) <= num_classes, "The encode number of classes must be larger than the max value of array."
    out = np.ones_like(array)*off_value
    out = np.expand_dims(out, axis=1)
    out = np.repeat(out, num_classes + 1, axis=1)
    
    for value in range(1, num_classes + 1):
        index_mask = array == value
        out[:, value, :, :][index_mask] = on_value

    return out


def DSC(A, B, epsilon=1e-5):
    A_area = np.sum(A)
    B_area = np.sum(B)
    AB_area = np.sum(A*B)

    result = (2 * AB_area + epsilon)/(A_area + B_area + epsilon)

    return result


def per_DSC(array_a, array_b, organ_amount):
    array_b = one_hot(array_b, organ_amount)
    result_list = []
    for organ_value in range(1, organ_amount + 1):
        result_per_organ = DSC(array_a[:, organ_value, :, :], array_b[:, organ_value, :, :])
        result_list.append(result_per_organ)

    return result_list


def avg_DSC(array_a, array_b, organ_amount):
    array_b = one_hot(array_b, organ_amount)
    result = 0
    for organ_value in range(1, organ_amount + 1):
        result_per_organ = DSC(array_a[:, organ_value, :, :], array_b[:, organ_value, :, :])
        result += result_per_organ
    
    return result/organ_amount


def volume_DSC(array_a, array_b, organ_amount):
    array_a = one_hot(array_a, organ_amount)
    array_b = one_hot(array_b, organ_amount)
    result_list = []
    for organ_value in range(1, organ_amount + 1):
        result_per_organ = DSC(array_a[:, organ_value, :, :], array_b[:, organ_value, :, :])
        result_list.append(result_per_organ)
    
    return [sum(result_list)/len(result_list)] + result_list


def JSC(A, B, epsilon=1e-5):
    A_area = np.sum(A)
    B_area = np.sum(B)
    AB_area = np.sum(A*B)

    result = (AB_area + epsilon) / (A_area + B_area - AB_area + epsilon)

    return result


def per_JSC(array_a, array_b, organ_amount):
    array_b = one_hot(array_b, organ_amount)
    result_list = []
    for organ_value in range(1, organ_amount + 1):
        result_per_organ = JSC(array_a[:, organ_value, :, :], array_b[:, organ_value, :, :])
        result_list.append(result_per_organ)

    return result_list


def avg_JSC(array_a, array_b, organ_amount):
    array_b = one_hot(array_b, organ_amount)
    result = 0
    for organ_value in range(1, organ_amount + 1):
        result_per_organ = JSC(array_a[:, organ_value, :, :], array_b[:, organ_value, :, :])
        result += result_per_organ
    
    return result/organ_amount


def volume_JSC(array_a, array_b, organ_amount):
    array_a = one_hot(array_a, organ_amount)
    array_b = one_hot(array_b, organ_amount)
    result_list = []
    for organ_value in range(1, organ_amount + 1):
        result_per_organ = JSC(array_a[:, organ_value, :, :], array_b[:, organ_value, :, :])
        result_list.append(result_per_organ)
    
    return [sum(result_list)/len(result_list)] + result_list


def calculate_surface_distances(array_a, array_b, spacing, connectivity=1):
    """
        reference url: https://mlnotebook.github.io/post/surface-distance-function/
    """
    array_a = np.atleast_1d(array_a.astype(np.bool))
    array_b = np.atleast_1d(array_b.astype(np.bool))

    conn = morphology.generate_binary_structure(array_a.ndim, connectivity)

    S = array_a ^ morphology.binary_erosion(array_a, conn)
    Sprime = array_b ^ morphology.binary_erosion(array_b, conn)

    distance_atob = morphology.distance_transform_edt(~S, spacing)
    distance_btoa = morphology.distance_transform_edt(~Sprime, spacing)
    
    total_surface_distances = np.concatenate([np.ravel(distance_atob[Sprime!=0]), np.ravel(distance_btoa[S!=0])])
       
    return total_surface_distances


def calculate_Hausdorff_distance(array_a, array_b, spacing):
    surf_dists = calculate_surface_distances(array_a, array_b, (spacing[2], spacing[0], spacing[1]))
    
    return np.max(surf_dists)


def calculate_robust_Hausdorff_distance(array_a, array_b, spacing):
    surf_dists = calculate_surface_distances(array_a, array_b, (spacing[2], spacing[0], spacing[1]))

    return np.percentile(surf_dists, 95)
    