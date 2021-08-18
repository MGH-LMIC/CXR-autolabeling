import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# Base model from DenseNet-121
class BaseModel(nn.Module):
    def __init__(self, in_dim=1, out_dim=31, name_model=None, pretrained=False, tf_learning=None):
        super().__init__()

        if name_model == None:
            self.main = tvm.densenet121(pretrained=True)
            self.main.features.conv0 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.classifier = nn.Linear(self.main.classifier.in_features, out_dim)
            if (pretrained==True):
                self.load_model(tf_learning)
        elif name_model == 'densenet161':
            self.main = tvm.densenet161(pretrained=True)
            self.main.features.conv0 = nn.Conv2d(in_dim, 96, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.classifier = nn.Linear(self.main.classifier.in_features, out_dim)
        elif name_model == 'resnet152':
            self.main = tvm.resnet152(pretrained=True)
            self.main.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.fc = nn.Linear(self.main.fc.in_features, out_dim)
        elif name_model == 'resnext101_32x8d':
            self.main = tvm.resnext101_32x8d(pretrained=True)
            self.main.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.main.fc = nn.Linear(self.main.fc.in_features, out_dim)
        else:
            print("this architecture is not supported")
            exit(-1)

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.main.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self.main.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    def load_model(self, path):
        # DataParallel make different layer names
        states = torch.load(path.resolve())
        new = list(states.items())
        my_states = self.main.state_dict()
        count=0
        for key, value in my_states.items():
            layer_name, weights = new[count]
            if my_states[key].shape == weights.shape:
                my_states[key] = weights
            else:
                print(f'pretrained weight skipping due to different shape: {key}')
            count+=1


    def forward(self, x):
        x = self.main(x)

        return x

