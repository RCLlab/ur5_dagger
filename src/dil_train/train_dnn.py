#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import copy
import os
import stream_tee as stream_tee
import __main__ as main
import csv
import random
torch.manual_seed(1)

class MyModel(nn.Module):
    def __init__(self, dev, input_dim, output_dim, hidden_dim, p):
        super().__init__()
        self.dev = dev
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim,hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim,output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
            x = self.dropout(x)
        out = self.layers[-1](x)
        return out

def train(epoch, dev, model, D_train, optimizer, loss_function, n_batch, total_len):
    runLoss = 0
    model.train()
    random.shuffle(D_train)
    x = np.array(D_train)[:, 0:48]
    y = np.array(D_train)[:, 48:54]
    for b in range(0, total_len, n_batch):
        seq_data = np.array(x[b:b+n_batch])
        seq_label = np.array(y[b:b+n_batch])
        seq_data = torch.tensor([i for i in seq_data], dtype=torch.float32).to(dev)
        seq_label = torch.tensor([i for i in seq_label], dtype=torch.float32).to(dev)
        optimizer.zero_grad()
        y_pred = model(seq_data)
        single_loss = loss_function(y_pred, seq_label)
        runLoss += single_loss.item()
        single_loss.backward()
        optimizer.step()
    runLoss /=total_len
    
    print('\nEpoch ', epoch)
    print ('Train loss: {:.10f} len {}'.format(runLoss, len(x)))
    
    return runLoss

def evals(model, D_eval, dev, loss_function, n_batch, total_len):
    total_loss = 0
    model.eval()
    random.shuffle(D_eval)
    x = np.array(D_eval)[:, 0:48]
    y = np.array(D_eval)[:, 48:54]
    with torch.no_grad():
        for b in range(0, total_len, n_batch):
            seq_data = np.array(x[b:b+n_batch])
            seq_label = np.array(y[b:b+n_batch])
            seq_data = torch.tensor([i for i in seq_data], dtype=torch.float32).to(dev)
            seq_label = torch.tensor([i for i in seq_label], dtype=torch.float32).to(dev)
            y_pred = model(seq_data)
            single_loss = loss_function(y_pred, seq_label)
            total_loss += single_loss.item()
        total_loss /= total_len
    
    print('Valid loss: {:.10f} len {}'.format(total_loss,len(x)))
    return total_loss


def load_data(n_files,datatype,filename):
    full_data = None
    for i in range(1, n_files):
        if datatype=='train':
            raw_data = np.loadtxt('{}_train_{}.csv'.format(filename,i), skiprows = 1, delimiter=',')
        else:
            if datatype=='eval':
                raw_data = np.loadtxt('{}_eval_{}.csv'.format(filename,i), skiprows = 1, delimiter=',')
            else:
                raw_data = np.loadtxt('{}_test_{}.csv'.format(filename,i), skiprows = 1, delimiter=',')
        if full_data is None:
            full_data = raw_data
        else:
            full_data = np.concatenate((full_data, raw_data), axis=0)
    return full_data


if __name__ == '__main__':
    run_name = stream_tee.generate_timestamp()
    main_dir = '/home/robot/workspaces/Big_Data/nn_log/'
    os.chdir(main_dir)
    os.makedirs(run_name)

    train_log = 'train_log.csv'
    eval_log = 'eval_log.csv'
    network_log = 'net_log.csv'
    filename = 'data_AB'
    if filename =='data_to_A':
        direction = '/home/user/workspaces/Big_Data/20220823_134405/'
        train_files = 4997
        eval_files = 1008
    if filename =='data_to_B':
        direction = '/home/user/workspaces/Big_Data/20220822_212339/'
        train_files = 3819
        eval_files = 678
    if filename =='data_AB':
        direction = '/home/robot/workspaces/Big_Data/mpc_log/AB_BA_acados'
        train_files = 4900
        eval_files = 607
    if filename =='data_BA':
        direction = '/home/robot/workspaces/Big_Data/mpc_log/AB_BA_acados'
        train_files = 4900
        eval_files = 607

    n_batch = 1000
    print(direction)
    os.chdir(direction)

    train_data = load_data(train_files,'train', filename)
    train_len = len(train_data)

    eval_data = load_data(eval_files,'eval', filename)
    eval_len = len(eval_data)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = main_dir+run_name
    os.chdir(log_dir)
    n = [48,36,27]
    drop = 0.25
    model = MyModel(dev,48,6,n,p=drop).to(dev)

    total_len = 15000
    for i in range(len(n)):
        with open(network_log,  'a') as fd:
            wr = csv.writer(fd, dialect='excel')
            wr.writerow([n[i],filename,'DNN', n_batch, drop, total_len])

    epoches = 500
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lower_loss = 1
    train_losses = []
    eval_losses = []
    D_train = []
    D_eval = []
    D_test = []

    for i in range(len(train_data)):
        D_train.append(train_data[i])
    
    for i in range(len(eval_data)):
        D_eval.append(eval_data[i]) 

    for epoch in range(epoches):
        log_dir = main_dir+run_name
        os.chdir(log_dir)
        
        loss1 = train(epoch, dev, model, D_train, optimizer, loss_function, n_batch, total_len)
        with open(train_log,  'a') as fd:
            wr = csv.writer(fd, dialect='excel')
            wr.writerow([loss1])
        
        loss2 = evals(model, D_eval, dev, loss_function, n_batch, total_len)
        with open(eval_log,'a') as fd:
            wr = csv.writer(fd, dialect='excel')
            wr.writerow([loss2])

        if loss2<lower_loss:
            lower_loss = loss2
            print('\nThe lowest loss is: {:4f}\n '.format(lower_loss))
            log_dir = main_dir+run_name
            os.chdir(log_dir)
            torch.save(model.state_dict(), 'model.pth')
    

