import torch
import numpy
import pandas
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,dim=768,head_num=8,drop1=0.,drop2=0.):
        super(Attention, self).__init__()
        self.linear = nn.Linear(dim,dim*3)
        self.W0 = nn.Linear(dim,dim)
        self.drop1 = nn.Dropout(drop1)
        self.drop2 = nn.Dropout(drop2)
        self.d = (dim/head_num)**-0.5
    def forward(self,x):
        # x:(batch,N,C)
        batch,N,C = x.shape
        qkv = self.linear(x) #(batch,N,3C)
        qkv = self.drop1(qkv)
        QKV = qkv.view(batch,N,3,8,-1)
        QKV = QKV.permute(2,0,3,1,4)
        q,k,v = QKV[0],QKV[1],QKV[2]
        attention = nn.functional.softmax((q@k.transpose(-1,-2))/self.d,dim=-1)
        attention = attention @ v
        attention = attention.transpose(1,2)  #torch.Size([64, 197, 8, 96])
        print(attention.shape)
        attention = attention.reshape(batch,N,C)
        attention = self.W0(attention)
        attention = self.drop2(attention)
        return attention

#%%
a=Attention()
x=torch.rand(64,197,768)
c=a(x)