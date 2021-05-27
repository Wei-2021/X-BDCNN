# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import glob
import cv2
import scipy.misc
from model import XBDCNN
from utils import Dataset, AddHybridNoise
import utils
import time
from tensorboardX import SummaryWriter
import csv
import codecs
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)


def DataAugmentation(temp_origin_img, temp_noise_img):
    if np.random.randint(2, size=1)[0] == 1:
        temp_origin_img = np.flip(temp_origin_img, axis=1)
        temp_noise_img = np.flip(temp_noise_img, axis=1)
    if np.random.randint(2, size=1)[0] == 1: 
        temp_origin_img = np.flip(temp_origin_img, axis=0)
        temp_noise_img = np.flip(temp_noise_img, axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        temp_origin_img = np.transpose(temp_origin_img, (1, 0, 2))
        temp_noise_img = np.transpose(temp_noise_img, (1, 0, 2))
    
    return temp_origin_img, temp_noise_img

def load_checkpoint(checkpoint_dir):
    if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
        # load existing model
        model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
        print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
        net = XBDCNN()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        epoch = model_info['epoch']
    else:
        # create model
        print("learning rate 1e-3")
        net = XBDCNN()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epoch = 0
    return model, optimizer, epoch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, checkpoint_dir + 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if epoch % lr_update_freq== 0 and epoch>1:
        for param_group in optimizer.param_groups:
            print("param_group['lr']",param_group['lr'],lr_update_freq)
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer



def data_write_csv(file_name, datas):  
    """write training log"""
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("file saved successfully")

if __name__ == '__main__':
    start_time = time.time()
    inputdir = '/home/yin/Code/ISBI2021/datasets/'
    checkpoint_dir = './checkpoint/'
    result_dir = './validation/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    assert os.path.exists(inputdir), "data not exist or data path is wrong"

    save_freq = 2
    lr_update_freq = 20
    epoches = 40
    batch_size = 64

    print('> Loading dataset ...',inputdir)
    dataset_train = Dataset(inputdir,train=True, shuffle=True)
    dataset_val = Dataset(inputdir,train=False, shuffle=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val,num_workers=0, batch_size=batch_size, shuffle=False)

    model, optimizer, state_epoch = load_checkpoint(checkpoint_dir)
    criterion = utils.fixed_loss()
    criterion = criterion.cuda()
    global_steps = 0
    val_step = 0
    writer = SummaryWriter() ## Visualization
    train_log = [[]]  ## write in csv
    val_log = [[]]  ## write in csv

    img = 0

    for epoch in range(state_epoch, epoches):
        start_epoch_time = time.time()

        optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)

        model.train()
        for i, data in enumerate(loader_train, 0):
            imgo_train = data
            img_train = imgo_train
            b,c,m,n = img_train.size()
            noiseimg = torch.zeros(img_train.size())
            noiselevel = torch.zeros(b,2,m,n)
            for nx in range(imgo_train.shape[0]):
                #########################
                noiseimg[nx,:,:,:], noiselevel[nx,:,:,:] = AddHybridNoise(img_train[nx, :, :, :].numpy())
                #########################
            input_var = Variable(noiseimg.cuda())  # noise image
            target_var = Variable(img_train.cuda()) #  clean image
            noise_level_var = Variable(noiselevel.cuda())
            # inference   
            target_res = input_var - target_var ### noise difference, label
            noise_level_est, output = model(input_var)  #noise level，noise residual
 
            loss = criterion(output, target_res, noise_level_est, noise_level_var, 1)
            #losses.update(loss.item())

            train_output = input_var - output #estimated denoised image

            psnr_train = utils.batch_psnr(train_output, target_var, 1.)

            global_steps += 1
            train_log.append([global_steps,loss.cpu().data.numpy(),psnr_train])
            writer.add_scalar('train_loss',loss,global_steps)
            writer.add_scalar('train_psnr',psnr_train,global_steps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("[Epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f Time: %.4f " \
                %(epoch+1, i+1, len(loader_train), loss.item(), psnr_train,time.time()-start_epoch_time))


        if epoch % save_freq==0 and epoch > 0:          
            for i, data in enumerate(loader_val, 0):
                val_time = time.time()

                imgo_val = data
                img_val = imgo_val
                noise_val = torch.zeros(img_val.size())
                b,c,m,n = noise_val.size()
                noiselevel_val = torch.zeros(b,2,m,n)
                for nx in range(imgo_val.shape[0]):
                #########################
                    noise_val[nx,:,:,:],noiselevel_val[nx,:,:,:] = AddHybridNoise(img_val[nx,:,:,:].numpy())
                #########################

                target_val, input_val = Variable(img_val.cuda()), Variable(noise_val.cuda())
                noiselevel_val = Variable(noiselevel_val.cuda())
                target_res = input_val - target_val  #label
                ##prediction
                with torch.no_grad():
                    noise_level_est, output = model(input_val)
                    
                    loss = criterion(output, target_res, noise_level_est, noiselevel_val, 1)
                
                out_val = input_val - output  
 
                psnr_val = utils.batch_psnr(out_val, target_val, 1.)
                print("VAL [Epoch %d][%d/%d] loss: %.4f PSNR_val: %.4f Time: %.4f " \
                    %(epoch+1, i+1, len(loader_val), loss.item(), psnr_train,time.time()-val_time))
                val_log.append([val_step,loss.cpu().data.numpy(),psnr_val])
                writer.add_scalar('val_loss',loss,val_step)
                writer.add_scalar('val_psnr',psnr_val,val_step)
                val_step += 1

                if img%100 ==0:
                    img_val_o= data.cpu().numpy()
                    img_val_o = img_val_o[0,:,:,:].transpose(2,1,0)
                    nimg_val = input_val.cpu().numpy()
                    nimg_val = nimg_val[0,:,:,:].transpose(2,1,0)
                    out_res = output.cpu().detach().numpy()
                    out_res = out_res[0,:,:,:].transpose(2,1,0)
                    pre_out = nimg_val - out_res
                    temp = np.concatenate((img_val_o,nimg_val, pre_out), axis=1)
                    cv2.imwrite(result_dir + '%04dimg_%d.png'%(epoch, img),np.clip(temp*255,0.0,255.0))

                img +=1  
                break

        save_checkpoint({'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}, is_best=0)       
            

    writer.close()
    data_write_csv('training_logs.csv',train_log)
    data_write_csv('validation_logs.csv',val_log)
    print("Training Done, total epoch: %d Time: %.4f"% (epoches,time.time()-start_time))
