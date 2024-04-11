import os
from typing import List
import inspect
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import AlexNet_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import timm

# from huggingface_hub import login, hf_hub_download


def get_image_net_preprocessor():
    return Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_alexnet() -> nn.Module:
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "alexnet", weights=AlexNet_Weights.DEFAULT
    )
    preprocess = get_image_net_preprocessor()
    return model, preprocess


def load_alexnet_softmax() -> nn.Module:
    model, preprocess = load_alexnet()
    model = nn.Sequential(model, nn.Softmax(dim=1))
    return model, preprocess


def load_alexnet_fc6() -> nn.Module:
    model, preprocess = load_alexnet()
    model.classifier = model.classifier[:2]
    return model, preprocess


def load_resnet18() -> nn.Module:
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    preprocess = get_image_net_preprocessor()
    return model, preprocess


def load_resnet18_softmax() -> nn.Module:
    model, preprocess = load_resnet18()
    model = nn.Sequential(model, nn.Softmax(dim=1))
    return model, preprocess


def load_dino2s() -> nn.Module:
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    preprocess = get_image_net_preprocessor()
    return dinov2_vits14, preprocess


def load_dino2l() -> nn.Module:
    dinov2_vitl14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    preprocess = get_image_net_preprocessor()
    return dinov2_vitl14, preprocess


def load_lunitdino():
    # load model from the hub
    model = timm.create_model(
        model_name="hf-hub:1aurent/vit_small_patch16_224.lunit_dino",
        pretrained=True,
    ).eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    model, transforms
 

def load_unidino2() -> nn.Module:
    # the model is assumed to be on your local drive
    local_dir = "/scratch/models/vit_large_patch16_224.dinov2.uni_mass100k/"
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(
        torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"),
        strict=True,
    )
    transform = Compose(
        [
            Resize(224),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    return model, transform


def get_loader_names() -> List[str]:
    def is_loader(name, obj):
        return (
            inspect.isfunction(obj)
            and name.startswith("load")
            and obj.__module__ == __name__
        )

    return [
        name
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if is_loader(name, obj)
    ]


def get_model(model_name: str) -> nn.Module:
    loader_name = f"load_{model_name}"
    if loader_name not in get_loader_names():
        raise ValueError(f"Model {model_name} not found.")
    loader = eval(loader_name)
    model, preprocess = loader()
    return model, preprocess


def get_availible_models() -> List[str]:
    loader_names = get_loader_names()
    model_names = [n[5:] for n in loader_names]
    return model_names
