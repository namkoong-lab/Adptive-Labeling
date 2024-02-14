#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np  


def l2_loss(dataloader_test, Predictor, device):
    res  = torch.empty((0), dtype=torch.float32, device=device)
    for (x_batch, label_batch) in dataloader_test:
        prediction = Predictor(x_batch) 
        diff_square = torch.square(torch.subtract(label_batch, prediction))
        res = torch.cat((res,diff_square),0) 
    return torch.mean(res)

 
