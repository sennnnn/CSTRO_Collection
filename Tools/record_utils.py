import os
import csv
import json

import pandas as pd
import SimpleITK as sitk

class TrainRecorder(object):
    def __init__(self, backup_path, save_key_word, loss_type, start_epoch=None):
        self.backup_path = backup_path
        self.save_key_word = save_key_word

        csv_record_path = os.path.join(backup_path, "{}@{}.csv".format(save_key_word, loss_type))
        txt_record_path = os.path.join(backup_path, "{}@{}.txt".format(save_key_word, loss_type))

        if not os.path.exists(backup_path):
            os.makedirs(backup_path, 0o777)

        if start_epoch != None and os.path.exists(csv_record_path):
            self.restore(start_epoch, csv_record_path)

        if os.path.exists(csv_record_path):
            self.csv_writer = csv.writer(open(csv_record_path, "a", newline=""))
        else:
            head_row = ["Epoch", "Loss", "Average_DSC", "Brain_Stem_DSC", "Eye_L_DSC", "Eye_R_DSC", "Lens_L_DSC", "Lens_R_DSC", "Optic_Nerve_L", "Optic_Nerve_R_DSC", "Optic_Chiasma_DSC", "Temporal_Lobes_L_DSC", "Temporal_Lobes_R_DSC", "Pituitary_DSC", "Parotid_Gland_L_DSC", "Parotid_Gland_R_DSC", \
    "Inner_Ear_L_DSC", "Inner_Ear_R_DSC", "Mid_Ear_L_DSC", "Mid_Ear_R_DSC", "Jaw_Joint_L_DSC", "Jaw_Joint_R_DSC", "Spinal_Cord_DSC", "Mandible_L_DSC", "Mandible_R_DSC"]
            self.csv_writer = csv.writer(open(csv_record_path, "w", newline=""))
            self.csv_writer.writerow(head_row)

        if os.path.exists(txt_record_path):
            self.txt_writer = open(txt_record_path, "a")
        else:
            self.txt_writer = open(txt_record_path, "w")

    def restore(self, start_epoch, csv_path):
        data = pd.read_csv(csv_path)
        data = data[data["Epoch"] < start_epoch]
        data.to_csv(csv_path, index=None)

    def write(self, row, log_string):
        self.csv_writer.writerow(self.row_norm(row))
        self.txt_writer.write(log_string + "\n")

    def row_norm(self, row):
        for index in range(len(row)):
            if isinstance(row[index], float):
                row[index] = round(row[index], 3)

        return row


class TestRecorder(object):
    def __init__(self, backup_path, save_key_word, epoch, performance, loss_type):
        self.backup_path = backup_path
        self.save_key_word = save_key_word

        self.json_record_path = os.path.join(backup_path, "{}@{}_{:.3f}_{}.json".format(save_key_word, epoch, performance, loss_type))
        self.test_result_folder = os.path.join(backup_path, "{}@{}_{:.3f}_{}_test".format(save_key_word, epoch, performance, loss_type))

        if not os.path.exists(self.test_result_folder):
            os.makedirs(self.test_result_folder, 0o777)

        self.json_writer = {}

    def write(self, patient, info):
        self.json_writer[patient] = info

    def patient_result_save(self, array, origin, spacing, patient):
        label = sitk.GetImageFromArray(array)
        label.SetOrigin(origin)
        label.SetSpacing(spacing)

        save_patient_folder = os.path.join(self.test_result_folder, patient)
        if not os.path.exists(save_patient_folder):
            os.makedirs(save_patient_folder, 0o777)
        sitk.WriteImage(label, os.path.join(save_patient_folder, "label.nii.gz"))

    def save(self):
        with open(self.json_record_path, "w") as fp:
            fp.write(json.dumps(self.json_writer, indent=4))
        