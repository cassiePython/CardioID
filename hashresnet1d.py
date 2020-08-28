from resnet1d import ResNet1D
import torch
import torch.nn as nn
import math

class HashResNet1D(ResNet1D):
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
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, hash_bit,
                 downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super().__init__(in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap,
                         increasefilter_gap, use_bn, use_do, verbose)
        self.hash_bit = hash_bit
        self.hash_layer = nn.Linear(512, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.dense = nn.Linear(hash_bit, n_classes)
        self.iter_num = 0
        self.__in_features = hash_bit
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale

    def forward(self, x):
        if self.training:
            self.iter_num += 1
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

        center_feature = self.hash_layer(out)
        if self.iter_num % self.step_size==0:
            self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
        code = self.activation(self.scale*center_feature)
        
        if self.verbose:
            print('hash layer', code.shape)
        out = self.dense(code)
        if self.verbose:
            print('dense', out.shape)

        return center_feature, code, out
