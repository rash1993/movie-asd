from __future__ import absolute_import

import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101, \
    resnet152
from torch.nn import Parameter


class ResNet(nn.Module):
    __factory = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=0, norm=False, embedding=True, dropout=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.embedding = embedding

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout

        out_planes = self.base.fc.in_features

        # Append new layers
        self.feat = nn.Linear(out_planes, self.num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.kaiming_normal_(self.feat.weight, mode='fan_out')
        init.constant_(self.feat.bias, 0)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.embedding:
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(x.size(0), -1)
            feature = self.feat(x)
            # feature = self.feat_bn(feature)
            if self.norm:
                # print(feature.size(), feature.norm(2,1).size())
                # print(feature.size())
                # print(feature.norm(2, 1).view(-1, 1).expand_as(feature))
                feature = feature / feature.norm(2, 1).view(-1, 1).expand_as(feature)
            if self.dropout > 0:
                feature = self.drop(feature)
        else:
            feature = x.view(x.size(0), -1)

        return feature

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def init_from_pretrained_model(self, pretrained_model):
        checkpoint = torch.load(pretrained_model)
        state_dict = checkpoint['state_dict']
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                          name, own_state[name].size(), param.size()))
                raise
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))
