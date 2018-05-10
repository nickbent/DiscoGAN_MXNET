#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 08:15:45 2018

@author: chocolatethunder
"""

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet import autograd
import numpy as np
from model import Discriminator, GeneratorCNN, network_init
from keras.preprocessing.image import ImageDataGenerator
from mxnet import ndarray as nd
import utils


class DiscoGAN(object):
    """
    Class for building the Disco GAN model
    """
    def __init__(self, learning_rate = 0.005, betas=(0.5, 0.999), conv_dim = 64):
        
        self.learning_rate = learning_rate
        self.betas = betas
        self.conv_dim = conv_dim
        self.build_model()
        

    def build_model(self):
        
        self.D_a = Discriminator(dim = self.conv_dim)
        self.D_b = Discriminator(dim = self.conv_dim)
        
        self.G_ab = GeneratorCNN(dim = self.conv_dim)
        self.G_ba = GeneratorCNN(dim = self.conv_dim)        
        
        network_init(self.D_a)
        network_init(self.D_b)
        network_init(self.G_ab)
        network_init(self.G_ba)
        
        
        self.chain_params()
        
        self.trainerD = gluon.Trainer(self.d, 'adam', {'learning_rate': self.learning_rate, 'beta1': self.betas[0], 'beta2': self.betas[1]})
        self.trainerG = gluon.Trainer(self.g, 'adam', {'learning_rate': self.learning_rate, 'beta1': self.betas[0], 'beta2': self.betas[1]})
            
        
        self.recon_criterion = gluon.loss.L2Loss()
        self.gan_criterion = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        self.feat_criterion = gluon.loss.HingeLoss()
        
        
    def chain_params(self):
            
        self.g = {}

        for key, value in self.G_ab.collect_params().items() :
            self.g[key] = value
        for key, value in self.G_ba.collect_params().items() :
            self.g[key] = value
            
        self.d = {}

        for key, value in self.D_a.collect_params().items() :
            self.d[key] = value
        for key, value in self.D_b.collect_params().items() :
            self.d[key] = value
    
    
    def data_loader( self, train_data_dir, batch_size = 256, img_height = 64, img_width = 64):
    
    # Initiate the train and test generators with data Augumentation 
    
        datagen = ImageDataGenerator(
                rescale = 1./255,
                fill_mode = "nearest")


        data_generator = datagen.flow_from_directory(
                train_data_dir,
                target_size = (img_height, img_width),
                batch_size = batch_size, 
                class_mode = None,
                shuffle = True)
    
        return(data_generator)
        
    def get_gan_loss(self, dis_real, dis_fake):
        
        #make sure this is the right axis
        labels_dis_real =  nd.ones(shape=(dis_real.shape[0], 1)) #Variable(torch.ones( [dis_real.size()[0], 1] ))
        labels_dis_fake = nd.zeros(shape=(dis_fake.shape[0], 1))#Variable(torch.zeros([dis_fake.size()[0], 1] ))
        labels_gen = nd.ones(shape=(dis_fake.shape[0], 1))#Variable(torch.ones([dis_fake.size()[0], 1]))

        dis_loss = self.gan_criterion( dis_real, labels_dis_real ) * 0.5 + self.gan_criterion( dis_fake, labels_dis_fake ) * 0.5
        gen_loss = self.gan_criterion( dis_fake, labels_gen )

        return dis_loss, gen_loss
    
    def discriminator_loss(self,dis_real, dis_fake):
            
        labels_dis_real =  nd.ones(shape=(dis_real.shape[0], 1)) 
        labels_dis_fake = nd.zeros(shape=(dis_fake.shape[0], 1))
        dis_loss = self.gan_criterion( dis_real, labels_dis_real ) * 0.5 + self.gan_criterion( dis_fake, labels_dis_fake ) * 0.5
        return dis_loss
    
    def gan_loss(self, dis_fake):
        labels_gen = nd.ones(shape=(dis_fake.shape[0], 1))
        gen_loss = self.gan_criterion( dis_fake, labels_gen )
        return gen_loss
        
        
        

    def get_fm_loss(self, real_feats, fake_feats):
        losses = 0
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            l2 = (real_feat.mean((0)) - fake_feat.mean((0))) * (real_feat.mean((0)) - fake_feat.mean((0)))
            loss = self.feat_criterion( nd.ones(shape=l2.shape), l2).mean()
            losses = losses +loss
        return losses
    


    def generate(self, a,b, path, save = True, epoch = None, num_images = 10, is_calc_loss = False  ):
            
        
        A = a.swapaxes(1,3).swapaxes(2,3)
        B = b.swapaxes(1,3).swapaxes(2,3)
        A, B = nd.array(A), nd.array(B)
        BA = self.G_ab( B )
        AB = self.G_ba( A )
        BAB = self.G_ba( BA )
        ABA = self.G_ab( AB )
        
        if is_calc_loss:
            gen_loss_A_total, gen_loss_B_total, dis_loss_A, dis_loss_B = self.calc_loss(A,B, BA, AB, BAB, ABA)
            
    
        
        
        if epoch is None :
            f_a = path+'A_Epoch_'
            f_b = path+'B_Epoch_'
        else :
            f_a =path +'A_'
            f_b =path +'B_'
     
        for im in range(num_images):      
            if save :
                            
                A_val = A[im]
                B_val = B[im]
                BA_val = BA[im]
                ABA_val = ABA[im]
                AB_val = AB[im]
                BAB_val = BAB[im]
                            
                filename = f_a+str(epoch)+'_im_'+str(im)+'.jpg'
                utils.save_image([A_val.asnumpy(),AB_val.asnumpy(),ABA_val.asnumpy()], filename = filename)
                filename = f_b+str(epoch)+'_im_'+str(im)+'.jpg'
                utils.save_image([B_val.asnumpy(),BA_val.asnumpy(),BAB_val.asnumpy()], filename = filename )
#            else:         
#                A_val = A[im].cpu().data.numpy() 
#                B_val = B[im].cpu().data.numpy() 
#                BA_val = BA[im].cpu().data.numpy()
#                ABA_val = ABA[im].cpu().data.numpy()
#                AB_val = AB[im].cpu().data.numpy()
#                BAB_val = BAB[im].cpu().data.numpy()
#                self.plt.close()
#                            
#                self.plt = self.show_images([A_val,AB_val,ABA_val], [B_val,BA_val,BAB_val])
        if is_calc_loss : 
            return gen_loss_A_total, gen_loss_B_total, dis_loss_A, dis_loss_B
    
    
    def calc_loss(self,x_A,x_B, x_BA, x_AB, x_BAB, x_ABA):
        
        A_dis_real, A_feats_real = self.D_a( x_A )
        A_dis_fake, A_feats_fake = self.D_a( x_BA )
    
        

        dis_loss_A = self.discriminator_loss( A_dis_real, A_dis_fake )

        B_dis_real, B_feats_real = self.D_b( x_B )
        B_dis_fake, B_feats_fake = self.D_b( x_AB )
        
        
        
        dis_loss_B = self.discriminator_loss( B_dis_real, B_dis_fake )

        recon_loss_A = self.recon_criterion( x_ABA, x_A )
        recon_loss_B = self.recon_criterion( x_BAB, x_B )
    

        A_dis_real, A_feats_real = self.D_a( x_A)
        A_dis_fake, A_feats_fake = self.D_a( x_BA )


        gen_loss_A = self.gan_loss( A_dis_fake )  


        fm_loss_A = self.get_fm_loss(A_feats_real, A_feats_fake)


        B_dis_real, B_feats_real = self.D_b( x_B )
        B_dis_fake, B_feats_fake = self.D_b( x_AB )

        gen_loss_B = self.gan_loss( B_dis_fake )  


        fm_loss_B = self.get_fm_loss(B_feats_real, B_feats_fake)
        rate = 0.01
        gen_loss_A_total = (gen_loss_B*0.1 + fm_loss_B*0.9) * (1.-rate) + recon_loss_A * rate
        gen_loss_B_total = (gen_loss_A*0.1 + fm_loss_A*0.9) * (1.-rate) + recon_loss_B * rate
                
    
        return gen_loss_A_total, gen_loss_B_total, dis_loss_A, dis_loss_B
        
        
    def train(self , n_epochs=100, batch_size=64, print_freq = 2, save = True, rate_tup = (0.01,0.5), 
              gan_curriculum = 1000,
              data_dir = '/home/chocolatethunder/Documents/Borealis/PyTorch/Disco_Gan/Data/facades/', 
              path = 'Generate_Images/'):
        
        train_dir_a = data_dir + 'train_A/'
        train_dir_b = data_dir + 'train_B/'
        
        val_dir_a = data_dir + 'val_A/'
        val_dir_b = data_dir + 'val_B/'
        
        path_train = path+'train'
        path_val = path+'val'
        
        val_A_generator = self.data_loader(val_dir_a, batch_size =  10)
        val_B_generator = self.data_loader(val_dir_b, batch_size = 10)
        
        self.G_ab.hybridize()
        self.G_ba.hybridize()
        self.D_a.hybridize()
        self.D_b.hybridize()
        
        
#        labels_real =  nd.ones(shape=(batch_size, 1)) 
#        labels_fake = nd.zeros(shape=(batch_size, 1))
#        
#        acc_a_tot = mx.metric.Accuracy()
#        acc_b_tot = mx.metric.Accuracy()
#        
#        acc_a_real = mx.metric.Accuracy()
#        acc_b_real = mx.metric.Accuracy()
#        
#        acc_a_fake = mx.metric.Accuracy()
#        acc_b_fake = mx.metric.Accuracy()
        
        for epoch in range(n_epochs):
            
            A_generator = self.data_loader(train_dir_a, batch_size = batch_size)
            B_generator = self.data_loader(train_dir_b, batch_size = batch_size)
            
            ctr = 0 
            for data_a, data_b in zip(A_generator, B_generator):    
                
                data_a = data_a.swapaxes(1,3).swapaxes(2,3)
                data_b = data_b.swapaxes(1,3).swapaxes(2,3)
                
                
                x_A, x_B = nd.array(data_a), nd.array(data_b)
                
                x_A.attach_grad()
                x_B.attach_grad()
                
                
                with autograd.record():


                    x_AB = self.G_ba(x_A)
                    x_BA = self.G_ab(x_B)
    
    
                    A_dis_real, A_feats_real = self.D_a( x_A )
                    A_dis_fake, A_feats_fake = self.D_a( x_BA )
                    

#                    acc_a_tot.update(preds = utils.binarize(temp), labels = labels_real)
#                    acc_a_real.update(preds = utils.binarize(A_dis_real), labels = labels_real)
#                    acc_a_tot.update(preds = utils.binarize(A_dis_fake), labels = labels_fake)
#                    acc_a_fake.update(preds = utils.binarize(A_dis_fake), labels = labels_fake)
                    
    
                    dis_loss_A = self.discriminator_loss( A_dis_real, A_dis_fake )

                    B_dis_real, B_feats_real = self.D_b( x_B )
                    B_dis_fake, B_feats_fake = self.D_b( x_AB )
                    
                    
#                    acc_b_tot.update(preds = utils.binarize(B_dis_real), labels = labels_real)
#                    acc_b_real.update(preds = utils.binarize(B_dis_real), labels = labels_real)
#                    acc_b_tot.update(preds = utils.binarize(B_dis_fake), labels = labels_fake)
#                    acc_b_fake.update(preds = utils.binarize(B_dis_fake), labels = labels_fake)
                    
                    
                    dis_loss_B = self.discriminator_loss( B_dis_real, B_dis_fake )
    
                    dis_loss = dis_loss_A + dis_loss_B
                    dis_loss.backward()
                self.trainerD.step(64)

                with autograd.record():


                    x_AB = self.G_ba(x_A)
                    x_BA = self.G_ab(x_B)
    

                    x_ABA = self.G_ab(x_AB)
                    x_BAB = self.G_ba(x_BA)


                    recon_loss_A = self.recon_criterion( x_ABA, x_A )
                    recon_loss_B = self.recon_criterion( x_BAB, x_B )
                
    
                    A_dis_real, A_feats_real = self.D_a( x_A)
                    A_dis_fake, A_feats_fake = self.D_a( x_BA )


                    gen_loss_A = self.gan_loss( A_dis_fake )  
    

                    fm_loss_A = self.get_fm_loss(A_feats_real, A_feats_fake)


                    B_dis_real, B_feats_real = self.D_b( x_B )
                    B_dis_fake, B_feats_fake = self.D_b( x_AB )
    
                    gen_loss_B = self.gan_loss( B_dis_fake )  
    

                    fm_loss_B = self.get_fm_loss(B_feats_real, B_feats_fake)
    
    
                    if ctr < gan_curriculum:
                        rate = rate_tup[0]
                    else:
                        rate = rate_tup[1]
                        
                    gen_loss_A_total = (gen_loss_B*0.1 + fm_loss_B*0.9) * (1.-rate) + recon_loss_A * rate
                    gen_loss_B_total = (gen_loss_A*0.1 + fm_loss_A*0.9) * (1.-rate) + recon_loss_B * rate
                
                    #total loss
                    gen_loss = gen_loss_A_total + gen_loss_B_total
                

                    gen_loss.backward()
                self.trainerG.step(64)
                
                if ctr % 4 == 0 :
                    print( "---------------------")
                    print ("GEN Loss:", gen_loss_A_total.asnumpy().mean(), gen_loss_B_total.asnumpy().mean())
                    print ("DIS Loss:", dis_loss_A.asnumpy().mean(), dis_loss_B.asnumpy().mean())
#                    print("Accuracy:", acc_a_tot.get()[1], acc_b_tot.get()[1])
#                    print("Accuracy of real:", acc_a_real.get()[1], acc_b_real.get()[1])
#                    print("Accuracy of fake:", acc_a_fake.get()[1], acc_b_fake.get()[1])
                
                if ctr == len(A_generator):
                    break
                ctr += 1
                
#            print( "---------------------")
#            print( "Accuracy of Epoch")
#            print("Accuracy:", acc_a_tot.get()[1], acc_b_tot.get()[1])
#            print("Accuracy of real:", acc_a_real.get()[1], acc_b_real.get()[1])
#            print("Accuracy of fake:", acc_a_fake.get()[1], acc_b_fake.get()[1])    
#            
#            acc_a_tot.reset()
#            acc_b_tot.reset()
#            acc_a_real.reset()
#            acc_b_real.reset()
#            acc_a_fake.reset()
#            acc_b_fake.reset()
                
            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                
                for data_a, data_b in zip(A_generator, B_generator):
                    self.generate(data_a, data_b, save = save, path = path_train, epoch = epoch)
                    break
                
                ctr = 0 
                
                gen_loss_a_mean = 0
                gen_loss_b_mean = 0
                dis_loss_a_mean = 0
                dis_loss_b_mean = 0
                for val_a, val_b in zip(val_A_generator, val_B_generator):
                    gen_loss_A_total, gen_loss_B_total, dis_loss_A, dis_loss_B = self.generate(val_a, val_b, 
                                            save = save, path = path_val, epoch = epoch, is_calc_loss = True)
                    gen_loss_a_mean +=gen_loss_A_total.asnumpy().mean()
                    gen_loss_b_mean +=gen_loss_B_total.asnumpy().mean()
                    dis_loss_a_mean +=dis_loss_A.asnumpy().mean()
                    dis_loss_b_mean +=dis_loss_B.asnumpy().mean()
                    
                    if ctr == len(val_A_generator):
                        break
                    ctr += 1

                print( "---------------------")
                print ("Val GEN Loss:", gen_loss_a_mean/ctr, gen_loss_b_mean/ctr)
                print ("Val DIS Loss:", dis_loss_a_mean/ctr, dis_loss_b_mean/ctr)
                
                
