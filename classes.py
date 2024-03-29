# Required class definitions here
from torch import nn
from torchvision.transforms.functional import center_crop
import torch 

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0):
        # What does super do?
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            # What does inplace do?
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.seq_block(x)
        return x 

class CNNBlocks(nn.Module):
    def __init__(self, n_conv, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0):
        # What does super do?
        super(CNNBlocks, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_conv):
            self.layers.append(CNNBlock(in_channels, out_channels, kernel_size, stride, padding))
            in_channels = out_channels 

    def forward(self, x):
        for layer in self.layers():
            x = layer(x)
        return x 

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, downhill=4):
        super(Encoder, self).__init__()

        self.enc_layers = nn.ModuleList()
        for _ in range(downhill):
            self.enc_layers += [
                CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding),
                nn.MaxPool2d(2, 2)
            ]
            in_channels = out_channels
            out_channels *= 2
        
        self.enc_layers.append(CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding))

    def forward(self, x):
        # Need to add skip connections
        skip_connection = []
        for layer in self.enc_layers():
            x = layer(x)
            if isinstance(layer, CNNBlocks):
                skip_connection.append(x)
        return x, skip_connection
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, exit_channels, padding, uphill=4):
        super(Decoder, self).__init__()

        self.dec_layers = nn.ModuleList()
        self.exit_channels = exit_channels

        for _ in range(uphill):
            self.dec_layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding),
            ]
            in_channels //= 2
            out_channels //= 2
        
        # Since CNNBlock has ReLU in the end, we need to add smth here
        self.dec_layers.append(nn.Conv2d(in_channels, exit_channels, kernel_size = 1, padding=padding))

    def forward(self, x, skip_connections):
        # Last layer is already connected
        skip_connections.pop(-1)

        for layer in self.dec_layers():
            if isinstance(layer, CNNBlocks):
                # Ideally center_crop shouldn't do anything here
                skip_connections[-1] = center_crop(skip_connections[-1], x.shape[2])
                # Concatenate across channels 
                x = torch.cat([x, skip_connections.pop(-1)], dim = 1)
            x = layer(x) 
        return x
            
class UNet(nn.Module):
    def __init__(self, in_channels, first_out_channels, exit_channels, downhill, padding = 0):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, first_out_channels,  padding, downhill)
        self.decoder = Decoder(first_out_channels*(2**downhill), first_out_channels*(2**(downhill - 1)), exit_channels, padding, downhill)

    def forward(self, x):
        x, skip = self.encoder(x)
        x = self.decoder(x, skip) 
        return x
