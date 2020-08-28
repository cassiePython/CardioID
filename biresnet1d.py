from resnet1d import ResNet1D
import torch
import torch.nn as nn

class BiResNet1D(ResNet1D):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        n_block: number of Bottleneck blocks
        n_classes: number of classes
        
    """
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, bits,
                 downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super().__init__(in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap,
                         increasefilter_gap, use_bn, use_do, verbose)
        self.bits = bits
        self.code = nn.Linear(512, self.bits)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(self.bits, n_classes)   

    def forward(self, x):
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every bottleneck has two conv
        for i_block in range(self.n_block):
            net = self.bottleneck_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        out = self.code(out)
        out = self.sigmoid(out)
        out = torch.round(out)
        #print (out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        #out = self.softmax(out)
        #if self.verbose:
        #    print('softmax', out.shape)
        
        return out
