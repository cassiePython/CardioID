"""
test on physionet data

Shenda Hong, Nov 2019
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from options.train_options import TrainOptions
from util import read_data_physionet, save_network
from resnet1d import ResNet1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary
from hashresnet1d import HashResNet1D
import loss
import os


if __name__ == "__main__":

    opt = TrainOptions().parse()

    is_debug = False
    n_classes = opt.n_classes
    batch_size = opt.batch_size
    window_size = opt.window_size
    stride = opt.stride
    hash_bit = opt.hash_bit
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    lr = opt.lr
    weight_decay = opt.w_decay
    weight_similarity = opt.w_similarity
    weight_classify = opt.w_classify
    n_epoch = opt.n_epoch
    
    if is_debug:
        writer = SummaryWriter('/home/wangcan/heartvoice/refer_code/log/debug')
    else:
        writer = SummaryWriter('/home/wangcan/heartvoice/refer_code/layer98_no_noise')

    # make data
    # preprocess_physionet() ## run this if you have no preprocessed data yet
    X_train, X_test, Y_train, Y_test = read_data_physionet(window_size=window_size, stride=stride, is_train=True)
    print(X_train.shape, Y_train.shape)
    dataset = MyDataset(X_train, Y_train, is_train=True)
    dataset_test = MyDataset(X_test, Y_test, is_train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False, shuffle=False)
    
    # make model
    #device_str = "cuda"
    #device = torch.device(device_str if torch.cuda.is_available() else "cpu")
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
    """
    model = ResNet1D(
        in_channels=1, 
        base_filters=128, # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=kernel_size, 
        stride=stride, 
        groups=32, 
        n_block=n_block, 
        n_classes=4, 
        downsample_gap=downsample_gap, 
        increasefilter_gap=increasefilter_gap, 
        use_do=True)
    """
    #model.to(device)
    model = model.cuda()

    #summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)
    # exit()

    # train and test
    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()
    center_loss_func = loss.CenterLoss(num_classes=n_classes, feat_dim=hash_bit, use_gpu=True)
    

    n_epoch = 20
    step = 0
    epoch = 0
    sigmoid_param = 10./hash_bit
    l_threshold = 15.0
    class_num = 1.0 # positive negative pairs balance weight
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):

            #input_x_one, input_y_one, input_x_two, input_y_two = tuple(t.to(device) for t in batch)
            input_x_one, input_y_one, input_x_two, input_y_two = tuple(t.cuda() for t in batch)

            inputs = torch.cat((input_x_one, input_x_two), dim=0)
            center_features, codes, outputs = model(inputs)

            #output_one = model(input_x_one)
            #output_two = model(input_x_two)

            #print (output_one)
            #print (output_two.shape)
            input_y_one_bi = torch.tensor(label_binarize(input_y_one, np.arange(n_classes))).cuda()
            input_y_two_bi = torch.tensor(label_binarize(input_y_two, np.arange(n_classes))).cuda()
            similarity_loss = loss.pairwise_loss(codes.narrow(0,0,input_x_one.size(0)), \
                                 codes.narrow(0,input_x_one.size(0),input_x_two.size(0)), \
                                 input_y_one_bi, input_y_two_bi, \
                                 sigmoid_param=sigmoid_param, \
                                 l_threshold=l_threshold, \
                                 class_num=class_num)
            center_loss = center_loss_func(center_features, torch.cat((input_y_one, input_y_two), dim=0))
            classify_loss_one = loss_func(outputs.narrow(0,0,input_x_one.size(0)), input_y_one)
            classify_loss_two = loss_func(outputs.narrow(0,input_x_one.size(0),input_x_two.size(0)), input_y_two)
            classify_loss = classify_loss_one + classify_loss_two
            #total_loss = similarity_loss + classify_loss + center_loss
            total_loss = classify_loss + center_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('Loss/similarity', similarity_loss.item(), step)
            writer.add_scalar('Loss/classify', classify_loss.item(), step)
            if is_debug:
                break
        epoch += 1
        network_label = 'hashnet'
        save_network(save_dir, model, network_label, epoch)
        scheduler.step(_)
            
        """
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            writer.add_scalar('Loss/train', loss.item(), step)

            if is_debug:
                break
        
        scheduler.step(_)
             
        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        ## vote most common
        final_pred = []
        final_gt = []
        for i_pid in np.unique(pid_test):
            tmp_pred = all_pred[pid_test==i_pid]
            tmp_gt = Y_test[pid_test==i_pid]
            final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
            final_gt.append(Counter(tmp_gt).most_common(1)[0][0])
        ## classification report
        tmp_report = classification_report(final_gt, final_pred, output_dict=True)
        print(confusion_matrix(final_gt, final_pred))
        f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] + tmp_report['3']['f1-score'])/4
        writer.add_scalar('F1/f1_score', f1_score, _)
        writer.add_scalar('F1/label_0', tmp_report['0']['f1-score'], _)
        writer.add_scalar('F1/label_1', tmp_report['1']['f1-score'], _)
        writer.add_scalar('F1/label_2', tmp_report['2']['f1-score'], _)
        writer.add_scalar('F1/label_3', tmp_report['3']['f1-score'], _)
        """  
