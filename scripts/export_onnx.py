"""Export the trained ResNet-34 emotion classifier to ONNX for GitHub Pages."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        backbone = models.resnet34(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_onnx.py INPUT.pth OUTPUT.onnx")
    input_path, output_path = map(Path, sys.argv[1:])
    model = EmotionClassifier()
    checkpoint = torch.load(input_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.zeros(1, 3, 160, 160), output_path,
        input_names=["image"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    print(f"Exported {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
