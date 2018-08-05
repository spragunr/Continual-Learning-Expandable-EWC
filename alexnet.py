import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, filters=64, classification_width=4096, num_classes=1000):
        """
        Constructor for AlexNet architecture of varying sizes.
        
        Sizes of layers not specified by parameters are automatically adjusted accordingly.

        Parameters:
        filters (int): number of convolutional filters applied by the first convolutional layer
                        (the out_channels from that layer)
        
        classification_width (int): the width of both of the linear (dense, fully connected) 
                                    layers used for mapping features to predictions
        
        num_classes (int): width of the output layer (predictions)- number of classes in 
                            classification problem
        """
        
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filters, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=filters, out_channels=filters*3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=filters*3, out_channels=filters*6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters*6, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels=256 * 6 * 6, out_channels=classification_width),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_channels=classification_width, out_channels=classification_width),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels=classification_width, out_channels=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
