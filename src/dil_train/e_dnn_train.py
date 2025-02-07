#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import os
import stream_tee as stream_tee
import __main__ as main
import csv
import random
torch.manual_seed(1)

class MyModel(nn.Module):
    def __init__(self, dev, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.dev = dev
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        # self.dropout = nn.Dropout(0.25)
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim,hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim,output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
            # x = self.dropout(x)
        out = self.layers[-1](x)
        return out

def train(epoch, dev, Encoder, Main_model, D_train, loss_function, Main_model_opt, n_batch, total_len):
    runLoss = 0
    Encoder.eval()
    Main_model.train()
    random.shuffle(D_train)
    robot_tr = np.array(D_train)[:, 0:6]
    human_tr = np.array(D_train)[:, 6:48]
    motion_tr = np.array(D_train)[:, 48:54]
    for b in range(0, total_len, n_batch):
        h_data = np.array(human_tr[b:b+n_batch])
        r_data = np.array(robot_tr[b:b+n_batch])
        m_data = np.array(motion_tr[b:b+n_batch])
        h_data = torch.tensor([i for i in h_data], dtype=torch.float32).to(dev)
        Main_model_opt.zero_grad()
        enc_out = Encoder(h_data)
        enc_out = enc_out.cpu().detach().numpy()
        temp = len(r_data)
        main_data = np.eye(temp, 30)
        main_data[:,0:6] = r_data
        main_data[:,6:30] = enc_out
        main_data = torch.tensor([i for i in main_data], dtype=torch.float32).to(dev)
        label = torch.tensor([i for i in m_data], dtype=torch.float32).to(dev)
        out = Main_model(main_data)
        single_loss = loss_function(label, out)
        runLoss += single_loss.item()
        single_loss.backward()
        Main_model_opt.step()
    runLoss /=total_len
    print ('Train epoch {} loss: {:.6f}'.format(epoch,  runLoss))
    
    return runLoss

def evals(Encoder, Main_model, D, dev, loss_function, n_batch, total_len):
    total_loss = 0
    Encoder.eval()
    Main_model.eval()
    random.shuffle(D)
    robot_tr = np.array(D)[:, 0:6]
    human_tr = np.array(D)[:, 6:48]
    motion_tr = np.array(D)[:, 48:54]
    n_batch = 1000
    with torch.no_grad():
        for b in range(0, total_len, n_batch):
            h_data = np.array(human_tr[b:b+n_batch])
            r_data = np.array(robot_tr[b:b+n_batch])
            m_data = np.array(motion_tr[b:b+n_batch])
            h_data = torch.tensor([i for i in h_data], dtype=torch.float32).to(dev)
            enc_out = Encoder(h_data)
            enc_out = enc_out.cpu().detach().numpy()
            temp = len(r_data)
            main_data = np.eye(temp, 30)
            main_data[:,0:6] = r_data
            main_data[:,6:30] = enc_out
            main_data = torch.tensor([i for i in main_data], dtype=torch.float32).to(dev)
            label = torch.tensor([i for i in m_data], dtype=torch.float32).to(dev)
            out = Main_model(main_data)
            single_loss = loss_function(label, out)
            total_loss += single_loss.item()
        total_loss /= total_len
    print('Evaluation Set: Average loss: {:4f}\n'.format(total_loss))
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
    method = 'E-O-DNN/' #'E-ODA-DNN' or 'E-O-DNN'
    
    main_dir = '/home/robot/workspaces/Big_Data/nn_weights/'
    os.chdir(main_dir)
    if not os.path.exists(method):
        os.makedirs(method)
    os.chdir(method)
    os.makedirs(run_name)
    
    train_log = 'train_log.csv'
    eval_log = 'eval_log.csv'
    test_log = 'test_log.csv'
    network_log = 'net_log.csv'

    filename ='data_BA'

    if filename =='data_to_A':
        direction = '/home/robot/workspaces/to_A/20220823_134405'
        train_files = 4900
        eval_files = 602
        test_files = 255
    if filename =='data_to_B':
        direction = '/home/robot/workspaces/to_B/20220822_212339'
        train_files = 3819
        eval_files = 698
        test_files = 310
    if filename =='data_AB':
        direction = '/home/robot/workspaces/AB_BA_acados'
        train_files = 5728
        eval_files = 1221
        test_files = 255
    if filename =='data_BA':
        direction = '/home/robot/workspaces/AB_BA_acados'
        train_files = 5734
        eval_files = 1215
        test_files = 255

    n_batch = 1000
    print(direction)
    os.chdir(direction)

    train_data = load_data(train_files,'train', filename)
    train_len = len(train_data)
    eval_data = load_data(eval_files,'eval', filename)
    eval_len = len(eval_data)

    test_data = load_data(test_files,'test',filename)
    test_len = len(test_data)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = main_dir+method+run_name
    os.chdir(log_dir)
    
    enc_layers = [48,36,24]
    n = [48,36,27]

    Encoder = MyModel(dev,42,enc_layers[2],enc_layers).to(dev)
    Encoder.load_state_dict(torch.load('/home/robot/workspaces/Big_Data/autoencoder/20220914_003323/model.pth', map_location=torch.device('cpu')))
    Main_model = MyModel(dev,30,6,n).to(dev)

    loss_function = nn.MSELoss()
    Enc_opt = torch.optim.Adam(Encoder.parameters(), lr=1e-3)
    Main_model_opt = torch.optim.Adam(Main_model.parameters(), lr=1e-3)
    lower_loss = 1

    total_len = 3000
    for i in range(len(n)):
        with open(network_log,  'a') as fd:
            wr = csv.writer(fd, dialect='excel')
            wr.writerow([n[i],filename, n_batch,total_len])

    epoches = 4000
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(Main_model.parameters(), lr=1e-3)
    lower_loss = 1
    train_losses = []
    eval_losses = []
    temp_mode = 0
    D_train = []
    D_eval = []
    D_test = []

    if method == 'E-ODA-DNN':
        temp_mode = 0
        nSteps = 300
        t_queue = total_len
        e_queue = total_len
        test_queue = total_len
        for i in range(total_len):
            D_train.append(train_data[i])
            D_eval.append(eval_data[i]) 
            D_test.append(test_data[i])
    else:
        for i in range(len(train_data)):
            D_train.append(train_data[i])
        for i in range(len(eval_data)):
            D_eval.append(eval_data[i]) 
        for i in range(len(test_data)):
            D_test.append(test_data[i])

    for epoch in range(epoches):
        if method == 'E-ODA-DNN':
            print(train_len, eval_len)
            print(len(D_train), len(D_eval), temp_mode)
            if temp_mode==0:
                if t_queue+nSteps<train_len:
                    for i in range(nSteps):
                        D_train.append(train_data[t_queue+i]) 
                    t_queue+=300
                temp_mode = 1
            else:
                if temp_mode==1:
                    if e_queue+nSteps<eval_len:
                        for i in range(nSteps):
                            D_eval.append(eval_data[e_queue+i]) 
                        e_queue+=300
                    temp_mode = 2
                else:
                    if temp_mode==2:
                        if test_queue+nSteps<test_len:
                            for i in range(nSteps):
                                D_test.append(test_data[test_queue+i]) 
                            test_queue+=300
                        temp_mode = 0

        os.chdir(log_dir)
        loss1 = train(epoch, dev, Encoder, Main_model, D_train, loss_function, optimizer, n_batch, total_len)
        with open(train_log,  'a') as fd:
            wr = csv.writer(fd, dialect='excel')
            wr.writerow([loss1])
        
        loss2 = evals(Encoder, Main_model, D_eval, dev, loss_function, n_batch, total_len)
        with open(eval_log,'a') as fd:
            wr = csv.writer(fd, dialect='excel')
            wr.writerow([loss2])

        test_loss = evals(Encoder, Main_model, D_test, dev, loss_function, n_batch, total_len)
        with open(test_log,'a') as fd:
            wr = csv.writer(fd, dialect='excel')
            wr.writerow([test_loss])

        if loss2<lower_loss:
            lower_loss = loss2
            print('\nThe lowest loss is: {:4f}\n '.format(lower_loss))
            os.chdir(log_dir)
            torch.save(Main_model.state_dict(), 'model.pth')
    

