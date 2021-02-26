#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import dgl
import espaloma as esp
# from torchdiffeq import odeint_adjoint as odeint


# In[3]:


dgl.__path__


# In[2]:


g = esp.Graph('C')


# In[3]:


layer = esp.nn.layers.dgl_legacy.gn()
representation = esp.nn.Sequential(layer, config=[32, "elu", 32, "elu"])
readout = esp.nn.readout.janossy.JanossyPooling(
    in_features=32, config=[32, "elu"],
    out_features={
        2: {'x': 32},
        3: {'x': 32},
    },
)
net = torch.nn.Sequential(
    representation,
    readout
)


# In[ ]:


net(g.heterograph)


# In[ ]:




