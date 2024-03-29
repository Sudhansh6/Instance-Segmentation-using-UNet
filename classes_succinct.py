from torch import nn
import torch 

class UNetBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, submodule, kernel=4, stride=2, padding=1, 
                 dropout = False, outermost = False, innermost = False):
        super(UNetBlock, self).__init__()

        down_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, stride=stride, padding=padding)
        down_batchnorm = nn.BatchNorm2d(mid_channels)
        down_relu = nn.ReLU(inplace=True)
        up_batchnorm = nn.BatchNorm2d(out_channels)
        up_relu = nn.ReLU(inplace=True)

        self.outermost = outermost 

        if outermost:
            layers = [down_conv, down_batchnorm, down_relu]
            layers += [submodule]
            layers += [nn.ConvTranspose2d(mid_channels*2, out_channels, kernel_size=kernel, stride=stride, padding=padding)]
            # ## Add final activation - segmentation uses sigmoid
            # layers += [nn.Sigmoid()]
            layers += [nn.Softmax(dim=1)]
        elif innermost:
            up_conv = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
            layers = [down_conv, down_batchnorm, down_relu]
            layers += [up_conv, up_batchnorm, up_relu]
        else: 
            up_conv = nn.ConvTranspose2d(mid_channels*2, out_channels, kernel_size=kernel, stride=stride, padding=padding)
            layers = [down_conv, down_batchnorm, down_relu]
            layers += [submodule]
            layers += [up_conv, up_batchnorm, up_relu]
            if dropout: layers += [nn.Dropout(0.4)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], axis = 1)

class UNet(nn.Module):
    # Exit channels is number of classes
    def __init__(self, in_channels, first_out_channels, exit_channels, downhill = 4, kernel = 3, stride = 1, padding = 0):
        super(UNet, self).__init__()
        
        num_innermost = first_out_channels*(2**(downhill - 1))
        num_mid = num_innermost
        layer = UNetBlock(num_innermost, num_innermost, num_innermost, None, innermost=True)
        for _ in range(downhill - 1):
            num_mid //= 2
            layer = UNetBlock(num_mid, num_mid*2, num_mid, layer)
            
        self.model = UNetBlock(in_channels, first_out_channels, exit_channels, layer, outermost=True)

    def forward(self, x):
        return self.model(x)
        