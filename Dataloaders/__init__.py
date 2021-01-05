from Dataloaders.CSTRO_dataset import CSTRO
from torch.utils.data import DataLoader


def construct_dataloader(params):
    if params["DATASET_SELECTION"] == "CSTRO":
        dataset = CSTRO(params)
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = params["BATCH_SIZE"],
            shuffle = params["IF_SHUFFLE"],
            num_workers = 0
        )
        
        return dataloader

