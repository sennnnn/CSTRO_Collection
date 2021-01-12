from Modeling.unet import UNet
from Modeling.attention_unet import AttentionUNet
from Modeling.ce_net import CENet
from Modeling.unet_plusplus import UNetPlusPlus
from Modeling.se_unet import SEUNet
from Modeling.r2u_net import R2UNet
from Modeling.wnet import WNet
from Modeling.attention_se_unet import AttentionSEUNet
from Modeling.attention_wnet import AttentionWNet
from Modeling.se_wnet import SEWNet


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
    elif "unet_plusplus" == params["MODEL_SELECTION"]:
        model = UNetPlusPlus(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model
    elif "se_unet" == params["MODEL_SELECTION"]:
        model = SEUNet(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model
    elif "r2u_net" == params["MODEL_SELECTION"]:
        model = R2UNet(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model
    elif "wnet" == params["MODEL_SELECTION"]:
        model = WNet(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model
    elif "attention_se_unet" == params["MODEL_SELECTION"]:
        model = AttentionSEUNet(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model
    elif "attention_wnet" == params["MODEL_SELECTION"]:
        model = AttentionWNet(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model
    elif "se_wnet" == params["MODEL_SELECTION"]:
        model = SEWNet(params["IN_CHANNELS"], params["ORGAN_AMOUNT"] + 1, params["BASE_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model