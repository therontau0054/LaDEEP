import torch
import torch.nn as nn
import torchvision.models as models

backbone_model = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152
}


class CrossSectionEncoder(nn.Module):
    def __init__(self, backbone_type = "resnet18"):
        super().__init__()
        resnet_model = backbone_model[backbone_type](weights = 'DEFAULT')
        modules = list(resnet_model.children())[:-1]
        self.section_encoder = nn.Sequential(*modules)
        self._freeze(self.section_encoder)
        self._make_adapter()

    def _make_adapter(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()

    def _forward_adapter(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out + identity
        return self.relu(out)

    def _freeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = torch.concat((x, x, x), dim = 1)
        x = self.section_encoder(x)
        x = x.view(-1, 1, 32, 16)
        return self._forward_adapter(x).view(-1, 512, 1, 1)


class CrossSectionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        hiddens = [512 // 2 ** i for i in range(8)]
        modules_front, modules_rear = [], []

        for i in range(len(hiddens) - 1):
            modules_front.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hiddens[i],
                        hiddens[i + 1],
                        kernel_size = 4,
                        stride = 2,
                        padding = 1
                    ),
                    nn.BatchNorm2d(
                        hiddens[i + 1]
                    ),
                    nn.ReLU())
            )
        modules_rear.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hiddens[-1],
                    hiddens[-1],
                    kernel_size = 4,
                    stride = 2,
                    padding = 1
                ),
                nn.BatchNorm2d(
                    hiddens[-1]
                ),
                nn.ReLU(),
                nn.Conv2d(
                    hiddens[-1],
                    out_channels = 1,
                    kernel_size = 1,
                    stride = 1
                ),
                nn.Sigmoid()
            )
        )
        self.decoder_front = nn.Sequential(*modules_front)
        self.decoder_rear = nn.Sequential(*modules_rear)

    def forward(self, x):
        x = self.decoder_front(x)
        x = nn.functional.interpolate(
            x,
            scale_factor = (2, 1),
            mode = "bilinear"
        )
        x = self.decoder_rear(x)
        return x
