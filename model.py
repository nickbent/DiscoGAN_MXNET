#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:03:37 2018

@author: chocolatethunder
"""
from mxnet.gluon import nn
import mxnet as mx

def deconv(channels, k_size = 4, stride=2, pad=1, bn=True, drop_out = True , p = 0.2 , ReLU = True, sequential = True):
    
    layers = []
    
    if ReLU :
        layers += [nn.Conv2DTranspose(channels=channels, strides = stride  , kernel_size=k_size, padding=pad, activation='relu', use_bias = False)]
    else: 
        layers += [nn.Conv2DTranspose(channels=channels, strides = stride  , kernel_size=k_size, padding=pad, use_bias = False)]
    if bn:
        layers += [nn.BatchNorm()]
    if drop_out :
        layers += [nn.Dropout(p)]
    if sequential :
        out = nn.HybridSequential()
        for layer in layers :
            out.add(layer)
        return out
            
    else:
        return layers

def conv(channels, k_size = 4, stride=2, pad=1, bn=True, drop_out = True , p = 0.2 , ReLU = True, sequential = True):
    
    layers = []
    
    if ReLU :
        layers += [nn.Conv2D(channels=channels, strides = stride  , kernel_size=k_size, padding=pad, activation='relu', use_bias = False)]
    else: 
        layers += [nn.Conv2D(channels=channels, strides = stride  , kernel_size=k_size, padding=pad, use_bias = False)]
    if bn:
        layers += [nn.BatchNorm()]
    if drop_out :
        layers += [nn.Dropout(p)]
    if sequential :
        out = nn.HybridSequential()
        for layer in layers :
            out.add(layer)
        return out
            
    else:
        return layers


def dense(channels, bn=True, drop_out = True , p = 0.2 , ReLU = True, sequential = True):
    
    layers = []
    
    if ReLU :
        layers += [nn.Dense(channels=channels, activation='relu')]
    else: 
        layers += [nn.Dense(channels=channels)]
    
    if bn:
        layers += [nn.BatchNorm()]
    if drop_out :
        layers += [nn.Droput(p)]
    
    if sequential :
        out = nn.HybridSequential()
        for layer in layers :
            out.add(layer)
        return out
            
    else:
        return layers

class DiscriminatorCNN(nn.HybridBlock):
    def __init__(self, dim = 64, num_layers = 5):
        super().__init__()
        
        
        with self.name_scope() : 
            layers = []
            layers += conv( dim, bn= False)

        
            for block in range(num_layers-2) :
                dim *= 2
                layers += conv( dim )
        
            layers += conv(1, stride=1, pad=0, bn= False, ReLU = False)
            
            self.model = nn.HybridSequential()
            with self.model.name_scope():
                for block in layers:
                    self.model.add(block)
    
    def hybrid_forward(self, F, x):
        
#        out = []
#        
#        for block in self.model :
#            x = block(x)
#            out += x
        
        return F.sigmoid(self.model(x))#, out[:-1]
    
    
class Discriminator(nn.HybridBlock):
    def __init__(self, dim = 64):
        super().__init__()
        
        
        with self.name_scope() : 

            self.conv1 = conv( dim, bn= False, ReLU = False)
            self.conv2 = conv( dim*2 , ReLU = False )
            self.conv3 = conv( dim*4 , ReLU = False)
            self.conv4 = conv( dim*8 , ReLU = False)
            self.conv5 = conv( 1 , stride=1, pad=0, bn = False , ReLU = False ) #,bn = False
        
    
    def hybrid_forward(self, F, x):
        
        x1 = F.relu(self.conv1(x))        
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = self.conv5(x4)
        
        #print('type(x): {}, F: {}'.format(type(x).__name__, F.__name__))
        return F.sigmoid(x5), [x2, x3, x4]



class GeneratorCNN(nn.HybridBlock):
    def __init__(self, dim = 64, channels = 3, num_layers = 8):
        super().__init__()
        
        with self.name_scope() :
            layers = []
        
            layers += conv(dim, bn= False)

            for layer in range(int(num_layers/2 -1)):
                dim *= 2
                layers += conv(dim)

            
            for layer in range(int(num_layers/2 -1)):
                dim = int(dim/2)
                layers += deconv(dim)

            
            layers += deconv(channels, bn= False)
            
            self.model = nn.HybridSequential()
            with self.model.name_scope():
                for block in layers:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        #print('type(x): {}, F: {}'.format(type(x).__name__, F.__name__))
        return F.sigmoid(self.model(x))

def param_init(param):
    if param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Uniform(0.02), ctx=mx.cpu())
        else:
            param.initialize(init=mx.init.Uniform(0.02), ctx=mx.cpu())
    elif param.name.find('batchnorm') != -1:
        
        if param.name.find('gamma') != -1:
            param.initialize(init=mx.init.Uniform(1), ctx=mx.cpu())
        else:
            param.initialize(init=mx.init.Zero(), ctx=mx.cpu())

def network_init(net):
    for param in net.collect_params().values():
        param_init(param)