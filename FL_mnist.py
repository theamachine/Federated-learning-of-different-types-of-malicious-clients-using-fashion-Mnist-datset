from py_interface import *
from ctypes import *

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import helper3 as helper
import seaborn as sns
import matplotlib.pyplot as plt

class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('client_num', c_int),
        ('max_data', c_int),
        ('data_id', c_int),
        ('validation_split', c_float)
    ]

class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('accuracy', c_float),
        ('epoch',c_int)
    ]

CLIENT_NUM = 20
DATA_SIZE = 20000
VALIDATION_SPLIT = 0.6
DATA_ID_DICT = {1:"cifar10",2:"fashionmnist"}
IMAGE_DIMENSION = [36,36]

PREFERED_IMAGE_SIZE = [36,36]

ns3Settings = {'client_num':CLIENT_NUM, 
               'max_data':DATA_SIZE,
               'validation_split':VALIDATION_SPLIT,
               'data_id':2}

mempool_key = 1234                                          
mem_size = 4096                                             
memblock_key = 2333  
print("starting!!")
exp = Experiment(mempool_key, mem_size, 'ELG_5142_NS3AI_Part3', '../../')
num_packet = 0
fl = Ns3AIRL(memblock_key, Env, Act)
flag = True

epochs = 10
local_epochs = 20
batch_size = 128

accuracy = 0

# SET SEED
seedNum = 3
torch.manual_seed(seedNum)
np.random.seed(seedNum)

############
is_malicious = [0 for i in range(CLIENT_NUM)]

partially_mal = 0.15
fully_mal = 0.05

num_partially_mal = int(CLIENT_NUM*partially_mal) # 1
num_fully_mal = int(CLIENT_NUM*fully_mal) # 2

cnt = 0
for i in range(num_partially_mal):
    is_malicious[cnt],cnt = 1,cnt+1
    
for i in range(num_fully_mal):
    is_malicious[cnt],cnt = 2,cnt+1

partially_split = 0.5
############

try:    
    for epoch in range(epochs):
        exp.reset()
        exp.run(setting=ns3Settings, show_output=True)
        with fl as data:
            if data == None:
                break
            data_name = DATA_ID_DICT[data.env.data_id]
            client_num = data.env.client_num
            data_size = data.env.max_data
            validation_split = data.env.validation_split

            # RETURN GLOBAL ACCURACY VALUE TO THE NETWORK
            data.act.epoch = epoch
            data.act.accuracy = accuracy
        
        print("epochs:{} i:{}".format(epochs,epoch+1))
        if flag:
            print("data to process: ",data_name)
            print("number of clients: ",client_num)
            print("data size: ",data_size)
            print("validation split: {:.2F}".format(validation_split))
            print("********************")

            print("Preparing Dataset...")
         
            classes, x_train, y_train,x_test, y_test = helper.get_fashionMnist_data()
            x_train, y_train = x_train[:DATA_SIZE], y_train[:DATA_SIZE]
            x_test, y_test = x_test[:int(len(x_test)*VALIDATION_SPLIT)], y_test[:int(len(y_test)*VALIDATION_SPLIT)]

            num_class = len(classes)
            client_list, client_data_split = helper.split_data(x_train,y_train,client_num,True,data_name,IMAGE_DIMENSION)
            
            print("Creating Model...")
            channel = x_train.shape[1]
            models = [helper.Net(num_class=num_class,num_channel=channel) for i in range(len(client_list))]
            if torch.cuda.is_available():
                models = [model.cuda() for model in models]
            client_model_split = {client:model for client,model in zip(client_list,models)}
            
            print("Training...")

            # Global Model
            global_model = helper.Net(num_class=num_class,num_channel=channel)
            
            flag = False
        
        client_validation_split, client_loss_split = {}, {}
        for client_name,mal in zip(client_list,is_malicious):
            # Syncronize with server's global model
            helper.syncronize_with_server_voter(global_model,client_model_split[client_name])

            data = client_data_split[client_name][0]
            label = client_data_split[client_name][1]
            model = client_model_split[client_name]

            ################
            if mal==1:
                half_shuffle_index = np.linspace(0,len(label)-1,len(label)).astype(int)
                h1 = half_shuffle_index[:int(len(half_shuffle_index)*partially_split)]
                h2 = half_shuffle_index[int(len(half_shuffle_index)*partially_split):]
                np.random.shuffle(h1)
                half_shuffle_index = np.concatenate((h1,h2),axis=0)

                label = label[half_shuffle_index] # half shuffled
            elif mal==2:
                shuffle_index = np.linspace(0,len(label)-1,len(label)).astype(int)
                np.random.shuffle(shuffle_index)
                label = label[shuffle_index] # # full shuffled
            ################

            client_model_split[client_name],loss = helper.train_local(model,data,label,client_name,local_epochs,batch_size)
            client_loss_split[client_name] = loss

            client_validation_split[client_name] = helper.validation(client_model_split[client_name],x_test,y_test,IMAGE_DIMENSION,data_name)[0]
            
            print("Validation Score {}:  {}".format(client_name,client_validation_split[client_name]))
            print("*******************************************************")

        ################
        losses = np.array([v for v in client_loss_split.values()])
        print("------------------------------------------------------------------")
        print(losses)
        print("------------------------------------------------------------------")
        sns.kdeplot(losses)
        plt.plot()
        ################

        # Aggregation
        print("Aggregation and validation!!!")
        helper.federated_averaging(global_model,client_model_split,client_loss_split)
        accuracy = helper.validation(global_model,x_test,y_test,IMAGE_DIMENSION,data_name)[0]
        print("Server validation Score: {}".format(accuracy))
        
        # Saving Model
        print("Model Downloading...")
        pickle.dump(global_model,open("models/global_model.pickle","wb"))

        if (epoch == epochs-1):
            print("Last Epoch: epoch {}:\taccuracy of global model is:{:.3F}".format(epoch+1,accuracy))
        
except Exception as e:
    print('Something wrong')
    print(e)
finally:
    del exp


