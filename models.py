import torch
import torch.nn as nn

cfg = {'small_VGG16': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],}
drop_rate = [0.3,0.4,0.4]

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        key = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.Dropout(drop_rate[key])]
                key += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ELU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
