import os
import torch


class Saver(object):
    def __init__(self, backup_path, save_key_word, loss_type, best_or_newest):
        self.backup_path   = backup_path
        self.save_key_word = save_key_word
        self.loss_type = loss_type
        self.best_or_newest = best_or_newest
        self.best_metric_value = -1
        
        if not os.path.exists(self.backup_path):
            os.makedirs(self.backup_path, 0o777)

    def restore(self):
        params_file_list = [x for x in os.listdir(self.backup_path) if os.path.splitext(x)[1] == ".pth"]
        params_file_list = [x for x in params_file_list if x.split("@")[0] == self.save_key_word]
        if(len(params_file_list) == 0):
            print(f"{self.backup_path} There are no model parameters file.")
            return None
        else:
            metric_extracter = lambda x: float(os.path.splitext(x)[0].split("@")[1].split("_")[1])
            epoch_extracter  = lambda x: int(os.path.splitext(x)[0].split("@")[1].split("_")[0])
            if self.best_or_newest == "best":
                value_table = [metric_extracter(x) for x in params_file_list]
            elif self.best_or_newest == "newest":
                value_table = [epoch_extracter(x) for x in params_file_list]

            selected_index = value_table.index(max(value_table))
            selected_params_file = params_file_list[selected_index]

            zip_file = torch.load(os.path.join(self.backup_path, selected_params_file))

            print(f"Load {selected_params_file}.")

            model_params = zip_file["model_params"]
            epoch        = zip_file["epoch"]
            performance  = zip_file["performance"]
            
            self.best_metric_value = performance

            return {"model_params": model_params, "epoch": epoch, "performance": performance}

    def save(self, epoch, metric_value, model):
        if metric_value > self.best_metric_value:
            save_name = "{}@{}_{:.3f}_{}.pth".format(self.save_key_word, epoch, metric_value, self.loss_type)
            save_path = os.path.join(self.backup_path, save_name)

            zip_file = {
                "model_params": model.state_dict(),
                "epoch"       : epoch,
                "performance" : metric_value,
            }
            torch.save(zip_file, save_path)

            print(f"Save {save_name}.")

            self.best_metric_value = metric_value