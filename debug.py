#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 05:29:37 2018

@author: chocolatethunder
"""

from model import *
from itertools import chain
from mxnet import gluon
import mxnet as mx
from mxnet import autograd


from mxnet import ndarray as nd
from DiscoGAN import DiscoGAN
DG = DiscoGAN()

batch_size=64
data_dir = '/home/chocolatethunder/Documents/Borealis/PyTorch/Disco_Gan/Data/facades/'
rate = 0.001

train_dir_a = data_dir + 'train_A/'
train_dir_b = data_dir + 'train_B/'


A_generator = DG.data_loader(train_dir_a, batch_size = batch_size)
B_generator = DG.data_loader(train_dir_b, batch_size = batch_size)

data_a = A_generator[0]
data_b = B_generator[0]

for data_a, data_b in zip(A_generator, B_generator):
    data_a = data_a.swapaxes(1,3).swapaxes(2,3)
    data_b = data_b.swapaxes(1,3).swapaxes(2,3)
    
    
    
    x_A = nd.array(data_a)
    x_B = nd.array(data_b)
    
    x_A.attach_grad()
    x_B.attach_grad()
    
    
    DG.G_ab.hybridize()
    DG.G_ba.hybridize()
    DG.D_a.hybridize()
    DG.D_b.hybridize()
    
    
    with autograd.record():
    
    
        x_AB = DG.G_ba(x_A)
        x_BA = DG.G_ab(x_B)
        
        
        A_dis_real, A_feats_real = DG.D_a( x_A )
        A_dis_fake, A_feats_fake = DG.D_a( x_BA )
        
        dis_loss_A = DG.discriminator_loss( A_dis_real, A_dis_fake )
    
        B_dis_real, B_feats_real = DG.D_b( x_B )
        B_dis_fake, B_feats_fake = DG.D_b( x_AB )
        
        dis_loss_B = DG.discriminator_loss( B_dis_real, B_dis_fake )
        
        dis_loss = dis_loss_A + dis_loss_B
    dis_loss.backward()
    DG.trainerD.step(batch_size = 64)
    
    with autograd.record():
    
    
        x_AB = DG.G_ba(x_A)
        x_BA = DG.G_ab(x_B)
        
    
        x_ABA = DG.G_ab(x_AB)
        x_BAB = DG.G_ba(x_BA)
    
    
        recon_loss_A = DG.recon_criterion( x_ABA, x_A )
        recon_loss_B = DG.recon_criterion( x_BAB, x_B )
                    
        
        A_dis_real, A_feats_real = DG.D_a( x_A)
        A_dis_fake, A_feats_fake = DG.D_a( x_BA )
    
    
        gen_loss_A = DG.gan_loss( A_dis_fake )  
        
    
        fm_loss_A = DG.get_fm_loss(A_feats_real, A_feats_fake)
    
    
        B_dis_real, B_feats_real = DG.D_b( x_B )
        B_dis_fake, B_feats_fake = DG.D_b( x_AB )
        
        gen_loss_B = DG.gan_loss( B_dis_fake )  
        
    
        fm_loss_B = DG.get_fm_loss(B_feats_real, B_feats_fake)
        
        gen_loss_A_total = (gen_loss_B*0.1 + fm_loss_B*0.9) * (1.-rate) + recon_loss_A * rate
        gen_loss_B_total = (gen_loss_A*0.1 + fm_loss_A*0.9) * (1.-rate) + recon_loss_B * rate
                    
                        #total loss
        gen_loss = gen_loss_A_total + gen_loss_B_total
                    
    
    gen_loss.backward()
    DG.trainerG.step(64)
    
    
    print ("GEN Loss:", gen_loss_A_total.asnumpy().mean(), gen_loss_B_total.asnumpy().mean())
    print ("DIS Loss:", dis_loss_A.asnumpy().mean(), dis_loss_B.asnumpy().mean())
    
    
    
from model import *
from itertools import chain
from mxnet import gluon
import mxnet as mx
from mxnet import autograd


from mxnet import ndarray as nd
from DiscoGAN import DiscoGAN
DG = DiscoGAN()

batch_size=64
data_dir = '/home/chocolatethunder/Documents/Borealis/PyTorch/Disco_Gan/Data/facades/'
rate = 0.01

train_dir_a = data_dir + 'train_A/'
train_dir_b = data_dir + 'train_B/'


A_generator = DG.data_loader(train_dir_a, batch_size = batch_size)
B_generator = DG.data_loader(train_dir_b, batch_size = batch_size)

data_a = A_generator[0]
data_b = B_generator[0]
data_a = data_a.swapaxes(1,3).swapaxes(2,3)
data_b = data_b.swapaxes(1,3).swapaxes(2,3)



x_A = nd.array(data_a)
x_B = nd.array(data_b)

x_A.attach_grad()
x_B.attach_grad()


DG.G_ab.hybridize()
DG.G_ba.hybridize()
DG.D_a.hybridize()
DG.D_b.hybridize()


with autograd.record():
    
    
    x_AB = DG.G_ba(x_A)
    x_BA = DG.G_ab(x_B)
    
    
    A_dis_real, A_feats_real = DG.D_a( x_A )
    A_dis_fake, A_feats_fake = DG.D_a( x_BA )
    
    dis_loss_A = DG.discriminator_loss( A_dis_real, A_dis_fake )
    
    B_dis_real, B_feats_real = DG.D_b( x_B )
    B_dis_fake, B_feats_fake = DG.D_b( x_AB )
    
    dis_loss_B = DG.discriminator_loss( B_dis_real, B_dis_fake )
    
    dis_loss = dis_loss_A + dis_loss_B

dis_loss.backward()
DG.trainerD.step(batch_size = 64)
print(A_dis_fake)
print(A_dis_real)


with autograd.record():


    x_AB = DG.G_ba(x_A)
    x_BA = DG.G_ab(x_B)
    

    x_ABA = DG.G_ab(x_AB)
    x_BAB = DG.G_ba(x_BA)


    recon_loss_A = DG.recon_criterion( x_ABA, x_A )
    recon_loss_B = DG.recon_criterion( x_BAB, x_B )
                
    
    A_dis_real, A_feats_real = DG.D_a( x_A)
    A_dis_fake, A_feats_fake = DG.D_a( x_BA )


    gen_loss_A = DG.gan_loss( A_dis_fake )

    fm_loss_A = DG.get_fm_loss(A_feats_real, A_feats_fake)


    B_dis_real, B_feats_real = DG.D_b( x_B )
    B_dis_fake, B_feats_fake = DG.D_b( x_AB )
    
    gen_loss_B = DG.gan_loss( B_dis_fake )  
    

    fm_loss_B = DG.get_fm_loss(B_feats_real, B_feats_fake)
    
    gen_loss_A_total = (gen_loss_B*0.1 + fm_loss_B*0.9) * (1.-rate) + recon_loss_A * rate
    gen_loss_B_total = (gen_loss_A*0.1 + fm_loss_A*0.9) * (1.-rate) + recon_loss_B * rate
                
                    #total loss
    gen_loss = gen_loss_A_total + gen_loss_B_total
                    
    
gen_loss.backward()
DG.trainerG.step(64)