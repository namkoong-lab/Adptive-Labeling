#!/usr/bin/env python
# coding: utf-8

# In[49]:


import torch
import numpy as np  
import higher


# In[223]:


res = 0
n = 5
h = torch.tensor([0.15 for i in range(n)])
c = torch.tensor([1, 0, 1, 0, 1])


tau = 0.1
gamma = 0.5
epsilon = 0.7

def sigmoid_gamma(x): #sigmoid func
    return 1/(1+ torch.exp(- gamma*x))
    
def approx_ber(h): #h is n-dim; output is an approx Bernoulli vector with mean h
    n = len(h)
    u = np.array([[np.random.uniform() for i in range(n)] for i in range(2)])
    G = torch.tensor(np.array([-np.log(-np.log(_)) for _ in u]))

    x1 = torch.exp((torch.log(h) + G[0])/tau)
    x2 = torch.exp((torch.log(torch.add(1,-h)) + G[1])/tau)
    x_sum = torch.add(x1,x2)
    x = torch.div(x1,x_sum)
    
    return x


def func(x):
    return approx_ber(x**2)


def Recall(h, c): #input is Bernoulli(h) and classifier c, output is recall
    Y_vec = approx_ber(h)
    n = len(h)
    
    #vec1 = sigmoid_gamma(torch.add(approx_ber(h), -epsilon))
    #vec2 = sigmoid_gamma(c)
    
    x = torch.sum(torch.mul(Y_vec,c))
    y = torch.sum(Y_vec)
    
    return x/y


# In[225]:


res = 0
n = 8
 
for j in range(100):
    x = torch.tensor([0.15 for i in range(n)], requires_grad = True)
    c = torch.bernoulli(torch.add(x,0.4))
    
    y = Recall(x,c) #approx_ber(x) #[0]#
    y.backward()
    print(x.grad)


# In[ ]:




