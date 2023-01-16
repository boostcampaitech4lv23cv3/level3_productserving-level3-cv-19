import io
from typing import List, Dict, Any

import albumentations
import albumentations.pytorch
from torchvision.transforms import Resize, Compose, CenterCrop

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
#from efficientnet_pytorch import EfficientNet
import timm

def ViT(num_classes):
    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    model.head = nn.Linear(in_features=1024, out_features=num_classes)
    return model


def get_model(model_path: str = "./assets/mask_task/model.pth") -> ViT:
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(num_classes=18).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def _transform_image(image_bytes: bytes) -> torch.Tensor:
    transform = albumentations.Compose([
            albumentations.Resize(height=512, width=384),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.2, 0.2, 0.2)),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image_array = np.array(image)
    return transform(image=image_array)['image'].unsqueeze(0)


def predict_from_image_byte(model: ViT, image_bytes: bytes, config: Dict[str, Any]) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        CenterCrop((320, 256)),
        Resize((224,224), Image.BILINEAR),
    ])
    tensor = _transform_image(image_bytes=image_bytes).to(device)
    tensor = transform(tensor)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return config["classes"][y_hat.item()]


def get_config(config_path: str = "./assets/mask_task/config.yaml"):
    import yaml

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
