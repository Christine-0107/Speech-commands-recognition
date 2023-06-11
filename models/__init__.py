from .vgg import *
from .wideresnet import *
from .dpn import *
from .resnet import *
from .densenet import *
from .resnext import *
from .resnet50 import *
from .resnet18_attention import *
from .resnet50_attention import *
from .vgg_attention import *
from .densenet_attention import DenseNet_A
from .resnext_attention import CifarResNeXt_A

available_models = [
    'vgg19_bn', 'vgg19_bn_attention',
    'resnet18', 'resnet18_attention', 'resnet50', 'resnet50_attention',
    'wideresnet28_10', 'wideresnet28_10D', 'wideresnet52_10',
    'resnext29_8_64', 'resnext29_8_64_attention', 
    'dpn92',
    'densenet_bc_100_12', 'densenet_bc_250_24', 'densenet_bc_190_40', 'densenet_bc_100_12_attention'
]

def create_model(model_name, num_classes, in_channels):
    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet18_attention":
        model = resnet18_attention(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet50":
        model = resnet50(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet50_attention":
        model = resnet50_attention(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "wideresnet28_10":
        model = WideResNet(depth=28, widen_factor=10, dropRate=0, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "wideresnet28_10D":
        model = WideResNet(depth=28, widen_factor=10, dropRate=0.3, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "wideresnet52_10":
        model = WideResNet(depth=52, widen_factor=10, dropRate=0, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnext29_8_64":
        model = CifarResNeXt(nlabels=num_classes, in_channels=in_channels)
    elif model_name == "resnext29_8_64_attention":
        model = CifarResNeXt_A(nlabels=num_classes, in_channels=in_channels)
    elif model_name == "dpn92":
        model = DPN92(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_100_12":
        model = DenseNet(depth=100, growthRate=12, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_100_12_attention":
        model = DenseNet_A(depth=100, growthRate=12, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_250_24":
        model = DenseNet(depth=250, growthRate=24, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_190_40":
        model = DenseNet(depth=190, growthRate=40, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "vgg19_bn":
        model = vgg19_bn(num_classes=num_classes, in_channels=in_channels)
    else:
        model = vgg19_bn_attention(num_classes=num_classes, in_channels=in_channels)
    return model
