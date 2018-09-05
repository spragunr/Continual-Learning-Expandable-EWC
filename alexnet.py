import torch.nn as nn
import torch.nn
import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, filters=8, num_classes=100):
        """
        Constructor for AlexNet architecture of varying sizes (For use with CIFAR).
        
        Sizes of layers not specified by parameters are automatically adjusted accordingly.

        Parameters:
        filters (int): number of convolutional filters applied by the first convolutional layer
                        (the out_channels from that layer)
        
        classification_width (int): the width of both of the linear (dense, fully connected) 
                                    layers used for mapping features to predictions
        
        num_classes (int): width of the output layer (predictions)- number of classes in 
                            classification problem
        """

        
        print(torch.randint(1, 10000, (1,), device=torch.device('cpu')))
        print(torch.randint(1, 10000, (1,), device=torch.device('cuda'))) 

        CLASSIFICATION_STARTING_WIDTH = 256
        CLASSIFICATION_SCALE_FACTOR = 256 
        
        FILTERS_START = 8
        FILTER_EXPANSION = 8 # TODO replace this with a pass-through of args.scale_factor

        super(AlexNet, self).__init__()
        

        self.filters = filters
       
        
        # scale dense layers' widths by CLASSIFICATION_SCALE_FACTOR each time filters expands
        # NOTE: now we're just adding, not multiplying by, CLASSIFICATION SCALE FACTOR
        classification_width = \
            (CLASSIFICATION_SCALE_FACTOR * ((filters - FILTERS_START) // FILTER_EXPANSION)) + CLASSIFICATION_STARTING_WIDTH 
        

        
        # TODO remove this - this is to prevent filters from expanding but maintain the 
        # expansion of the classification width correctly 
        # filters = FILTERS_START

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filters, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=filters, out_channels=filters*3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=filters*3, out_channels=filters*6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters*6, out_channels=filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters*4, out_channels=filters*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=filters*4, out_features=classification_width),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=classification_width, out_features=classification_width),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=classification_width, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
