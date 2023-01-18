import io
from os import path, getcwd
import yaml
import torch
import torch.nn as nn
import torchvision.models as models

from PIL import Image
from typing import Tuple
from torchvision.transforms import Resize, ToTensor, Compose, Normalize, CenterCrop, InterpolationMode
from torchvision.models import EfficientNet_B0_Weights, ResNet152_Weights, EfficientNet_B3_Weights

prj_dir = getcwd()


class ModelMask(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ModelAge(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ResNet152 = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.ResNet152.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.ResNet152(x)
        return x


class ModelGender(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def transform_image(image_bytes) -> torch.Tensor:
    transform = Compose([
        CenterCrop((720, 540)),
        Resize((256, 192), InterpolationMode.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
    ])

    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')

    return transform(image).unsqueeze(0)


def get_prediction(model, image_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return tensor, y_hat


def load_model(model_type: str):
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if model_type == "model_mask":
        model = ModelMask(num_classes=3).to(device)
    elif model_type == "model_age":
        model = ModelAge(num_classes=3).to(device)
    elif model_type == "model_gender":
        model = ModelGender(num_classes=2).to(device)
    model.load_state_dict(torch.load(config['model_path'][model_type], map_location=device))

    return model


def get_config(config_path: str = path.join(prj_dir, "assets/config.yaml")):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg
