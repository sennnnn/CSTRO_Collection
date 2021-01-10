import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import argparse

from Dataloaders import *
from Dataloaders.preprocess import *
from Modeling import *

from Tools.loss_utils import *
from Tools.metric_utils import *
from Tools.lr_utils import *
from Tools.ckpt_utils import *
from Tools.record_utils import *
from Tools.utils import *


def train(params):
    # Data Loader
    dataloader           = construct_dataloader(params)
    total_step_per_epoch = len(dataloader)
    # Inference Related Things
    model        = construct_model(params)
    loss         = construct_loss(params)
    optimizer    = torch.optim.Adam([{"params": model.parameters(), "lr": params["LR"]}])
    lr_scheduler = LrScheduler("poly", params["LR"], params["TOTAL_EPOCHES"], total_step_per_epoch, params["WARM_UP_EPOCHES"])
    # Ckpt Tools
    save_tool    = Saver(params["BACKUP_PATH"], params["CUSTOM_KEY"], params["LOSS_TYPE"], "best")    
    zip_file     = save_tool.restore()
    if zip_file != None:
        load_state            = model.load_state_dict(zip_file["model_params"])
        params["START_EPOCH"] = zip_file["epoch"] + 1
    # Record Tools
    record_tool = TrainRecorder(params["BACKUP_PATH"], params["CUSTOM_KEY"], params["LOSS_TYPE"], params["START_EPOCH"])
    # Main Loop
    for epoch in range(params["START_EPOCH"] - 1, params["TOTAL_EPOCHES"]):
        loss_epoch_avg = 0
        DSC_epoch_avg  = 0
        for step, batch in enumerate(dataloader):
            # Inference
            input_data = batch["image"]
            label      = batch["label"]
            predict, predict_softmax = model(input_data)
            loss_variable = loss(predict, label)
            # Learning Rate Schedule
            lr_scheduler(optimizer, step + 1, epoch + 1)
            # Gradient Back Propogation
            optimizer.zero_grad()
            loss_variable.backward()
            optimizer.step()
            # To CPU
            logit  = predict_softmax.data.cpu().numpy()
            target = label.cpu().numpy()
            # Loss Value and Metric Value
            loss_value          = loss_variable.item()
            DSC_value           = avg_DSC(logit, target, params["ORGAN_AMOUNT"])
            DSC_value_per_organ = per_DSC(logit, target, params["ORGAN_AMOUNT"])
            loss_epoch_avg      += loss_value
            DSC_epoch_avg       += DSC_value
            # Log Information Print
            loss_string   = "Loss: {:+.3f}".format(loss_value)
            metric_string = "DSC: Average {:+.3f}".format(DSC_value)
            log_string    = "Epoch: {:<3} Step: {:>4}/{:<4} {} {}".format(epoch + 1, step + 1, total_step_per_epoch, loss_string, metric_string)
            print(log_string)
            # Record Middle Result
            record_tool.write([epoch + 1, loss_value, DSC_value, *tuple(DSC_value_per_organ)], log_string)

        loss_epoch_avg /= total_step_per_epoch
        DSC_epoch_avg  /= total_step_per_epoch
        params["TASK"] = "test"
        total_result = test(params, model, DSC_epoch_avg, epoch + 1)
        params["TASK"] = "train"
        save_tool.save(epoch + 1, total_result["DSC"][0], model)


def test(params, model=None, train_performance=None, epoch=None):
    test_src_folder = params["TEST_SRC_FOLDER"]
    patient_list    = os.listdir(test_src_folder)
    # Inference Related Things
    if model == None:
        model       = construct_model(params)
        # Ckpt Tools
        save_tool   = Saver(params["BACKUP_PATH"], params["CUSTOM_KEY"], params["LOSS_TYPE"], "best")    
        zip_file    = save_tool.restore()
        load_state  = model.load_state_dict(zip_file["model_params"])
        performance = zip_file["performance"]
        epoch       = zip_file["epoch"]
        model       = model.eval()
    else:
        model       = model
        performance = train_performance
        epoch       = epoch
    # Preprocess
    tr_list = [
        CropResize(params["CROP_RATE"], params["INPUT_SIZE"]),
        WindowCenterWidth(params["WINDOW_CENTER"], params["WINDOW_WIDTH"]),
        Normalize(),
        ToTensor(params["CUDA"]),
    ]
    # Record Tools
    record_tools = TestRecorder(params["BACKUP_PATH"], params["CUSTOM_KEY"], epoch, performance, params["LOSS_TYPE"])
    # Main Loop
    total_result        = {}
    total_result["DSC"] = [0 for i in range(params["ORGAN_AMOUNT"] + 1)]
    for patient in patient_list:
        patient_folder = os.path.join(test_src_folder, patient)
        main_image, origin, spacing = read_NiFTI(os.path.join(patient_folder, "data.nii.gz"))
        main_label = read_array_from_NiFTI_path(os.path.join(patient_folder, "label.nii.gz"))
        dataloader = [{"image": x, "label": y} for x, y in  zip(main_image, main_label)]
        # Inference
        seg_map_list = []
        with torch.no_grad():
            n, h, w = main_label.shape
            for batch in dataloader:
                for tr in tr_list:
                    batch = tr(batch)
                input_data = batch["image"].unsqueeze(dim=0)
                label      = batch["label"]
                predict, predict_softmax = model(input_data)
                logit = predict_softmax.data.cpu().numpy()
                seg_map = np.argmax(logit, axis=1).astype(np.uint8)
                seg_map = pad_resize(seg_map[0], params["CROP_RATE"], (h, w))
                seg_map_list.append(seg_map)
        seg_map_array = np.array(seg_map_list)
        DSC_result = volume_DSC(main_label, seg_map_array, params["ORGAN_AMOUNT"])
        result = {}
        result["DSC"] = DSC_result
        total_result["DSC"] = [total_result["DSC"][index] + DSC_result[index] for index in range(params["ORGAN_AMOUNT"] + 1)]
        Metric_string = "{:.14}: \n".format(patient)
        Metric_string += "DSC: Average {:+.3f}".format(DSC_result[0])
        print(Metric_string)
        record_tools.patient_result_save(seg_map_array, origin, spacing, patient)
        record_tools.write(patient, result)

    total_result["DSC"] = [total_result["DSC"][index]/len(patient_list) for index in range(params["ORGAN_AMOUNT"] + 1)]
    Metric_string = "{:.14}: \n".format("Average")
    Metric_string += "DSC: Average {:+.3f}".format(total_result["DSC"][0])
    print(Metric_string)
    record_tools.write("Average", total_result)
    record_tools.save()

    return total_result


if __name__ == "__main__":
    params = {}

    # Data load settings
    params["BATCH_SIZE"] = 4
    params["IF_SHUFFLE"] = True
    params["DATASET_SELECTION"] = "CSTRO"
    params["ROOT_FOLDER"] = "Data/train_flatten"
    params["TEST_SRC_FOLDER"] = "Data/test"
    # Data augmentation settings
    params["CROP_RATE"] = (0.3125, 0.75, 0.3125, 0.75)
    params["INPUT_SIZE"] = (224, 224)
    params["WINDOW_CENTER"] = 50
    params["WINDOW_WIDTH"] = 350
    params["ROTATE_ANGEL"] = 15
    params["TRANSLATE_RANGE"] = 15
    # Model settins
    params["IN_CHANNELS"] = 1
    params["ORGAN_AMOUNT"] = 22
    params["BASE_CHANNELS"] = 64
    # Task settings
    params["TASK"] = "train"
    params["LR"] = 0.001
    params["CUDA"] = True
    params["START_EPOCH"] = 1
    params["WARM_UP_EPOCHES"] = 1
    params["TOTAL_EPOCHES"] = 100
    params["LOSS_TYPE"] = "FOCAL"
    params["MODEL_SELECTION"] = "unet"
    # Other settings
    params["GPU_VALID_NUMBER"] = "0"
    params["CUSTOM_KEY"] = f"{params['MODEL_SELECTION']}"
    params["BACKUP_PATH"] = f"Backup/{params['CUSTOM_KEY']}"

    os.environ["CUDA_VISIBLE_DEVICES"] = params["GPU_VALID_NUMBER"]

    if params["TASK"]   == "train":
        train(params)
    elif params["TASK"] == "valid":
        valid(params)
    elif params["TASK"] == "test":
        test(params)