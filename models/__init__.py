from .DRSN import *
from .ResNet import *
from .LSTM import *
from .MobileNetV3 import *



def load_model(name, num_classes=X, **kwargs):

    model_dict = globals()
    model = model_dict[name](num_classes=num_classes, **kwargs)
    return model
