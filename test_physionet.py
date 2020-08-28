"""
test on physionet data

Can Wang, Aug 2020
"""

import numpy as np
from tqdm import tqdm
import time
from util import *
from resnet1d import ResNet1D, MyDataset
from options.test_options import TestOptions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary
from hashresnet1d import HashResNet1D
from collections import Counter
import loss
import dill
import os

if __name__ == "__main__":

    opt = TestOptions().parse()

    is_debug = False
    n_classes = opt.n_classes
    batch_size = opt.batch_size
    window_size = opt.window_size
    stride = opt.stride
    hash_bit = opt.hash_bit
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    load_step = opt.load_epoch

    # make data
    # preprocess_physionet() ## run this if you have no preprocessed data yet
    X_train, X_test, Y_train, Y_test = read_data_physionet(window_size=window_size, stride=stride, is_train=False)
    print("Train Set:", X_train.shape, Y_train.shape)
    print("Test Set:", X_test.shape, Y_test.shape)
    dataset = MyDataset(X_train, Y_train, is_train=False)
    dataset_test = MyDataset(X_test, Y_test, is_train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False, shuffle=False)

    kernel_size = 16
    stride = 2
    n_block = 48
    downsample_gap = 6
    increasefilter_gap = 12
    
    model = HashResNet1D(
        in_channels=1, 
        base_filters=64, # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=kernel_size, 
        stride=stride,
        groups=32, 
        n_block=n_block, 
        n_classes=n_classes,
        hash_bit=hash_bit,
        downsample_gap=downsample_gap,
        increasefilter_gap=increasefilter_gap, 
        use_do=True)
    model = model.cuda()
     
    network_label = 'hashnet'
    load_network(save_dir, model, network_label, load_step)

    # train and test
    model.verbose = False
    model.eval()

    SigBankPath = os.path.join(save_dir, 'SigBank.pkl')

    if not os.path.exists(SigBankPath):    
        SigBank = [] # [(subject,hashcode),...,(subject,hashcode)]
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter):

                input_x, input_y = tuple(t.cuda() for t in batch)
                _, codes, outputs = model(input_x)
                codes = torch.sign(codes)
                #print (input_x.shape[0])
                for ind in range(input_x.shape[0]):
                    subject = int(input_y[ind].cpu().data.numpy())
                    hashcode = codes[ind].cpu().data.numpy()
                    SigBank.append((subject,hashcode))
                    #print ((subject,hashcode))
        SigBank = sorted(SigBank, key=lambda x:x[0])
        # remove duplicate
        SigBank = remove_dup(SigBank)       
        # save SigBank
        res = {'SigBank':SigBank}
        with open(SigBankPath, 'wb') as fout:
            dill.dump(res, fout)

    with open(SigBankPath,'rb') as fin:
        res = dill.load(fin)
    SigBank = res['SigBank']

    TestCodesPath = os.path.join(save_dir, 'TestCodes.pkl')
       
    if not os.path.exists(TestCodesPath):
        TestCodes = []
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):

                input_x, input_y = tuple(t.cuda() for t in batch)
                _, codes, outputs = model(input_x)
                codes = torch.sign(codes)
                for ind in range(input_x.shape[0]):
                    subject = int(input_y[ind].cpu().data.numpy())
                    hashcode = codes[ind].cpu().data.numpy()
                    TestCodes.append((subject,hashcode))
        # save TestCodes
        TestCodes = sorted(TestCodes, key=lambda x:x[0])
        res = {'TestCodes':TestCodes}
        with open(TestCodesPath, 'wb') as fout:
            dill.dump(res, fout)

    with open(TestCodesPath,'rb') as fin:
        res = dill.load(fin)
    TestCodes = res['TestCodes']

    # This is a very naive(simple) version to search the SigBank - Just use to test the code
    # Its efficiency is unsatisfactory. But it can be accelerated by using 'early stop' described in the paper.
    # Other parallel processing strategies can also be used
    # Due to some engineering techniques, code of this part can not be available currently.
    TestCodesDic = conv_TestCodes(TestCodes)
    top_k = 100
    most = 20
    total = 0
    start = time.time()
    for k, v in TestCodesDic.items():
        total_subjects = pred_one(v, SigBank, top_k)      
        total_subjects = Counter(total_subjects)
        result = total_subjects.most_common(most)
        s = [item[0] for item in result]
        if k in s:
            total += 1
            #print (total)
    end = time.time()
    print("Time :%f秒"%(end-start))
    print (total)
    print (len(TestCodesDic))
        
        

    
    
    

    
           
    
