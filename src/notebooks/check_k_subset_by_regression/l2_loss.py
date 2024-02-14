#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np  


def l2_loss(fnet, dataloader_test, Predictor, device, para):
    res  = torch.empty((0), dtype=torch.float32, device=device)
    for (x_batch, label_batch) in dataloader_test:
        prediction = Predictor(x_batch) 
        diff_square = torch.square(torch.subtract(label_batch, prediction))
        res = torch.cat((res,diff_square),0) 
    return torch.mean(res)

 

def var_l2_loss_estimator(fnet, dataloader_test, Predictor, device, para):
    z_dim = para['z_dim']
    N_iter =  para['N_iter']
    if_print =  para['if_print']

    res  = torch.empty((0), dtype=torch.float32, device=device)
    res_square  = torch.empty((0), dtype=torch.float32, device=device)

    
    for i in range(N_iter):
        z_pool = torch.randn(z_dim, device=device)# sample z
        l2_loss_list = torch.empty((0), dtype=torch.float32, device=device)
        for (x_batch, label_batch) in dataloader_test:
            fnet_logits = fnet(x_batch, z_pool)  #enn output
            prediction = Predictor(x_batch)  #prediction output

            l2_list_temp = torch.square(torch.subtract(fnet_logits, prediction)) #l2 los

            l2_loss_list = torch.cat((l2_loss_list,l2_list_temp),0) 
        #print("i:",i)
        l2_est = torch.mean(l2_loss_list) #avg l2 loss
        #print("recall_est:", recall_est)
        res = torch.cat((res,(l2_est).view(1)),0)
        #print("res:",res)
        res_square = torch.cat((res_square,(l2_est ** 2).view(1)),0)
        
    #print("res_square:", res_square)
    var = torch.mean(res_square) - (torch.mean(res)) ** 2
    if if_print == 1:
        print('l2 list', res)
        print("var of l2_loss:",var)
        print("mean of l2_loss",  torch.mean(res))
    return var