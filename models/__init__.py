from .DRSN import *
from .resnet import *
from .LSTM import *
from .MobileNetV3 import *
from .resnet_teacher import *
from .MobileNetV3_teacher import *
from .DRSN_teacher import *


# def load_model(name, num_classes=4, pretrained=False, **kwargs):
#
#     model_dict = globals()
#     model = model_dict[name](pretrained=pretrained, num_classes=num_classes, **kwargs)
#     return model
def load_model(name, num_classes=5, **kwargs):

    model_dict = globals()
    model = model_dict[name](num_classes=num_classes, **kwargs)
    return model