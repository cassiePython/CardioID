import numpy as np
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import os
import torch

def preprocess_physionet():
    """
    download the raw data from https://physionet.org/content/challenge-2017/1.0.0/, 
    and put it in challenge2017/
    """
    
    # read label
    label_df = pd.read_csv('challenge2017/REFERENCE-v3.csv', header=None)
    label = label_df.iloc[:,1].values
    print(Counter(label))

    # read data
    all_data = []
    filenames = pd.read_csv('challenge2017/training2017/RECORDS', header=None)
    filenames = filenames.iloc[:,0].values
    print(filenames)
    for filename in tqdm(filenames):
        mat = scipy.io.loadmat('challenge2017/training2017/{0}.mat'.format(filename))
        mat = np.array(mat['val'])[0]
        all_data.append(mat)
    all_data = np.array(all_data)

    res = {'data':all_data, 'label':label}
    with open('challenge2017/challenge2017.pkl', 'wb') as fout:
        pickle.dump(res, fout)

def slide_and_cut(X, Y, window_size, stride, output_pid=False):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        # This aims to 'augment' or 'balance' the data,
        # You can use according to your dataset or just set 'i_stride = stride'
        """
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            i_stride = stride//6 # use 10 for read_data_physionet_2
        elif tmp_Y == 2:
            i_stride = stride//2
        elif tmp_Y == 3:
            i_stride = stride//20
        """
        i_stride = stride
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)

def read_data_physionet(window_size=2000, stride=1000, is_train=True):

    # read pkl
    with open('challenge2017/challenge2017.pkl', 'rb') as fin:
        res = pickle.load(fin)
    ## scale data
    all_data = res['data']
    all_label = res['label']
    new_data = []
    new_label = []
    # exclude noise 
    for i in range(len(all_label)):
        if all_label[i] != '~':
            new_data.append(all_data[i])
            new_label.append(all_label[i])
    all_data = np.array(new_data)
    new_label = np.array(new_label)     
    
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    ## encode label
    all_label = []
    for i in new_label:
        if i == 'N':
            all_label.append(0)
        elif i == 'A':
            all_label.append(1)
        elif i == 'O':
            all_label.append(2)
        elif i == '~':
            all_label.append(3)
    all_label = np.array(all_label)
    
    # slide and cut
    X, _, Y = slide_and_cut(all_data, all_label, window_size=window_size, stride=stride, output_pid=True)
  
    # split train test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    # shuffle train
    if is_train:
        shuffle_pid = np.random.permutation(Y_train.shape[0])
        X_train = X_train[shuffle_pid]
        Y_train = Y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    return X_train, X_test, Y_train, Y_test

def save_network(save_dir, network, network_label, epoch_label):
    save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.state_dict(), save_path)
    print ('saved net: %s' % save_path)

def load_network(save_dir, network, network_label, epoch_label):
    load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
    load_path = os.path.join(save_dir, load_filename)
    assert os.path.exists(
        load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

    network.load_state_dict(torch.load(load_path))
    print ('loaded net: %s' % load_path)

def ttest_onesided(s, T):
    from scipy.stats import ttest_1samp
    (t, p) = ttest_1samp(s, T)
    if t > 0:
        onesided_p = 1 - p / 2
    else:
        onesided_p = p / 2
    return onesided_p, p

# engineering techniques can be used to improve the efficiency
def remove_dup(SigBank):
    last = 0
    curr = 1
    length = len(SigBank)
    while curr != length:
        item_last = SigBank[last]
        item_curr = SigBank[curr]
        if item_curr[0] == item_last[0] and (item_curr[1] == item_last[1]).all():
            del SigBank[curr]
            length -= 1
        else:
            last += 1
            curr += 1
        
    return SigBank

def similarity(code_one, code_two):
    # convert -1 to 0
    code_one = 1 * (code_one > 0) * code_one
    code_two = 1 * (code_two > 0) * code_two
    # Hamming distance
    smstr=np.nonzero(code_one-code_two)
    sm = np.shape(smstr[0])[0]
    return sm

def search_all(TestCodes, SigBank, top_k):
    output = []
    for item in TestCodes:
        dist_table = search_one(item, SigBank)
        # fetch top_k
        dist_top_k = dist_table[:top_k]
        output.append((real_subject,[s[0] for s in dist_top_k]))
    return output
        
def search_one(hashcode, SigBank):
    dist_table = []
    for sig in SigBank:
        subject = sig[0]
        code = sig[1]
        dist = similarity(hashcode, code)
        dist_table.append((subject, dist))
    dist_table = sorted(dist_table, key=lambda x:x[1])
    return dist_table

def conv_TestCodes(TestCodes):
    TestCodesDic = {}
    for item in TestCodes:
        subject = item[0]
        code = item[1]
        if subject not in TestCodesDic:
            TestCodesDic[subject] = [code]
        else:
            TestCodesDic[subject].append(code)
    return TestCodesDic

def pred_one(hashcodes, SigBank, top_k):
    total_subjects = []
    for hashcode in hashcodes:
        dist_table = search_one(hashcode, SigBank)
        dist_table = dist_table[:top_k]
        target_subjects = list(set([item[0] for item in dist_table]))
        total_subjects += target_subjects
    #total_subjects = Counter(total_subjects)
    #total.most_common(1)
    return total_subjects
    
        
