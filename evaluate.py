import os
import json
import shutil
import argparse

import pandas as pd
import numpy as np

from Tools.metric_utils import calculate_surface_distances, JSC, DSC
from Tools.utils import parse_list, read_NiFTI, list_avg, list_stdev


def evaluate(model_name, ground_truth_folder, metric_name="DSC", backup_path="Backup"):
    model_backup_path = "{}/{}".format(backup_path, model_name)

    path_list = os.listdir(model_backup_path)
    pth_list = [x for x in path_list if os.path.splitext(x)[1] == ".pth" ]
    pth_list = sorted(pth_list, key=lambda x: float(x.split("@")[1].split("_")[1]), reverse=True)
    folder_list = [x for x in path_list if os.path.isdir(os.path.join(model_backup_path, x))]
    # folder_list = sorted(folder_list, key=lambda x: int(x.split("@")[1].split("_")[0]), reverse=True)

    assert len(pth_list) != 0

    best_model_epoch = pth_list[0].split("@")[1].split("_")[0]
    best_model_test_result_folder = [x for x in folder_list if x.split("@")[1].split("_")[0] == best_model_epoch]
    best_model_test_result_folder = os.path.join(model_backup_path, best_model_test_result_folder[0])
    test_patient_list = os.listdir(ground_truth_folder)

    result_storage_folder = "Material/Result/Json/"

    if not os.path.exists(result_storage_folder):
        os.makedirs(result_storage_folder, 0o777)

    result_storage_path = f"{result_storage_folder}/{model_name}@{best_model_epoch}.json"
    if os.path.exists(result_storage_path):
        fp = open(result_storage_path, "r")
        metric = json.load(fp)
    else:
        metric = {}

    collect = {i: [] for i in range(1, 23)}

    for patient in test_patient_list:
        if patient not in metric.keys():
            metric[patient] = {}
        metric[patient][metric_name] = []
        predict_patient_folder = os.path.join(best_model_test_result_folder, patient)
        label_patient_folder = os.path.join(ground_truth_folder, patient)
        predict_path = os.path.join(predict_patient_folder, "label.nii.gz")
        ground_truth_path = os.path.join(label_patient_folder, "label.nii.gz")

        img_array, origin, spacing = read_NiFTI(os.path.join(label_patient_folder, "data.nii.gz"))
        predict_array, _, _ = read_NiFTI(predict_path)
        ground_truth_array, _, _ = read_NiFTI(ground_truth_path)

        string = "{:<14} ".format(patient)

        for i in range(1, 23):
            organ_predict_array = (predict_array == i).astype(np.uint8)
            organ_ground_truth_array = (ground_truth_array == i).astype(np.uint8)
            if metric_name == "DSC":
                metric_value = DSC(organ_predict_array, organ_ground_truth_array)
            elif metric_name == "Jaccard":
                metric_value = JSC(organ_predict_array, organ_ground_truth_array)
            elif metric_name == "HD":
                surf_dists = calculate_surface_distances(organ_predict_array, organ_ground_truth_array, (spacing[2], spacing[0], spacing[1]))
                max_hd = np.max(surf_dists)
                robust_hd = np.percentile(surf_dists, 95)
                avg_sd = np.average(surf_dists)
                metric_value = {"max HD": max_hd, "95% HD": robust_hd, "avg SD": avg_sd}
            metric[patient][metric_name].append(metric_value)
            collect[i].append(metric_value)
            if isinstance(metric_value, dict):
                for key in metric_value.keys(): 
                    string += "{}:{:.3f} ".format(key, metric_value[key])
            else:
                string += "{}:{:.3f} ".format(metric_name, metric_value)

        print(string)

    if metric_name != "HD":
        string = "{:<14} ".format("Average")
        for i in range(1, 23):
            string += "{:.3f} ".format(list_avg(collect[i]))
        print(string)
        string = "{:<14} ".format("Stdev")
        for i in range(1, 23):
            string += "{:.3f} ".format(list_stdev(collect[i]))
        print(string)

    with open(result_storage_path, "w") as fp:
        fp.write(json.dumps(metric, indent=4))


def collect_training_validation_curves(model_name, backup_path):
    model_backup_path = "{}/{}".format(backup_path, model_name)

    result_storage_folder = "Material/Training/Json"

    if not os.path.exists(f"{result_storage_folder}/{model_name}"):
        os.makedirs(f"{result_storage_folder}/{model_name}", 0o777)

    collect = {}
    # with open(f"{result_storage_folder}/{model_name}_validation.txt", mode="w") as fp:
    item_list = os.listdir(model_backup_path)
    json_list = [x for x in item_list if os.path.splitext(x)[1] == ".json"]
    json_list = sorted(json_list, key = lambda x: int(x.split("@")[1].split("_")[0]))

    count = 0
    for item in json_list:
        item_path = os.path.join(model_backup_path, item)
        count += 1
        with open(item_path, "r") as f:
            metric = json.load(f)
            collect[count] = metric['Average']['DSC'][0]
    
    with open(f"{result_storage_folder}/{model_name}/{model_name}_validation.json", mode="w") as fp:
        fp.write(json.dumps(collect, indent=4))

    collect = {}
    csv_path = [x for x in item_list if os.path.splitext(x)[1] == ".csv"][0]
    csv_path = os.path.join(model_backup_path, csv_path)

    csv = pd.read_csv(csv_path)
    epoch_list = list(set(csv["Epoch"]))

    for epoch in epoch_list:
        collect[epoch] = np.average(np.array(csv["Loss"][csv["Epoch"] == epoch]))

    with open(f"{result_storage_folder}/{model_name}/{model_name}_training.json", mode="w") as fp:
        fp.write(json.dumps(collect, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--test_folder", type=str, default=r"Data/test")
    parser.add_argument("--metric_name", type=str, default=None)
    parser.add_argument("--validation_extraction", type=bool, default=False)

    args = parser.parse_args()

    if args.validation_extraction:
        collect_training_validation_curves(args.model_name, "Backup")
    else:
        evaluate(args.model_name, args.test_folder, args.metric_name, "Backup")
    