import torch 
import torch.nn as nn

def conv3_3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    '''
    3x3 convolution with padding
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


def conv1_1(in_planes, out_planes, stride=1):
    '''
    1x1 convolution
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_c, out_c, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_c * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv1_1(in_c, width)
        # self.bn1 = norm_layer(width)
        # self.conv2 = conv3_3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)
        # self.conv3 = conv1_1(width, out_c * self.expansion)
        # self.bn3 = norm_layer(out_c * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        # self.stride = stride

        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=width, kernel_size=1, stride= 1, bias=False)
        self.bn1 = norm_layer(width)
        #------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, 
                               kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.bn2 = norm_layer(width)
        #------------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_c * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(out_c * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        pass 


class ResNet(nn.Module):

    def __init__(self, block, layers, num_class=1000, zero_init_residual=False, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_c = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None " \
                             f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_c, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_c)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)


    def _make_layer(self, block, channel, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        dowmsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_c != channel * block.expansion:
            dowmsample = nn.Sequential(
                nn.Conv2d(self.in_c, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_c, channel, stride, dowmsample, self.groups, 
                            self.base_width, previous_dilation, norm_layer))
        self.in_c = channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_c, channel, groups=self.groups, base_width=self.base_width, 
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        pass

    def forward(self, x):
        pass


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet50(**kwargs):
    ''' 
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    '''
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    ''' 
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    '''
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)