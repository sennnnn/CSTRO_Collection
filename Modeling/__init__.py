from Modeling.unet import UNet
from Modeling.attention_unet import AttentionUNet
from Modeling.ce_net import CENet


def construct_model(params):
    if "unet" == params["MODEL_SELECTION"]:
        model = UNet(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model
    elif "attention_unet" == params["MODEL_SELECTION"]:
        model = AttentionUNet(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model
    elif "ce_net" == params["MODEL_SELECTION"]:
        model = CENet(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model