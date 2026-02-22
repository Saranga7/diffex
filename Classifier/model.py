import torch
import torch.nn as nn
from torchvision import models


class PretrainedClassifier(nn.Module):
    def __init__(self, pretrain = True):
        super(PretrainedClassifier, self).__init__()
        self.model = models.mobilenet_v2(pretrained = True)

        if not pretrain:
            for param in self.model.parameters():
                param.requires_grad = False

        # Make the last layer have only 2 outputs instead of 1000.
        self.model.classifier[1] = nn.Linear(1280, 2)

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    model = PretrainedClassifier()

    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(output, output.shape)

    print("\n\n")


    