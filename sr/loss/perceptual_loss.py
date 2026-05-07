import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:35])  # relu4_2
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, input_img, target_img):
        if input_img.shape[1] == 1:
            input_img = input_img.repeat(1, 3, 1, 1)
            target_img = target_img.repeat(1, 3, 1, 1)
        input_features = self.vgg_layers(input_img)
        target_features = self.vgg_layers(target_img)
        return torch.mean((input_features - target_features) ** 2)
