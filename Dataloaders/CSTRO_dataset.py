import os
import torch

from Dataloaders.preprocess import *
from torch.utils.data import Dataset


class CSTRO(Dataset):
    def __init__(self, params):
        self.params = params

        self.task = params["TASK"]
        self.if_cuda = params["CUDA"]
        self.root_folder = params["ROOT_FOLDER"]
        self.item_list = os.listdir(self.root_folder)

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        item_path = os.path.join(self.root_folder, self.item_list[index])

        item = np.load(item_path)
        image = item["image"]
        label = item["label"]

        data_pack = {"image": image, "label": label}

        data_pack = self._preprocess(data_pack)

        return data_pack

    def _preprocess(self, data):
        params = self.params

        if self.task == "test" or self.task == "valid":
            tr_list = [
                CropResize(params["CROP_RATE"], params["INPUT_SIZE"]),
                WindowCenterWidth(params["WINDOW_CENTER"], params["WINDOW_WIDTH"]),
                Normalize(),
                ToTensor(self.if_cuda),
            ]
        elif self.task == "train":
            tr_list = [
                CropResize(params["CROP_RATE"], params["INPUT_SIZE"]),
                WindowCenterWidth(params["WINDOW_CENTER"], params["WINDOW_WIDTH"]),
                Normalize(),
                RandomRotate(params["ROTATE_ANGEL"]),
                RandomTranslate(params["TRANSLATE_RANGE"]),
                ToTensor(self.if_cuda),
            ]
        else:
            assert False, "{} underdefined task type.".format(self.task)

        for tr in tr_list:
            data = tr(data)

        return data        