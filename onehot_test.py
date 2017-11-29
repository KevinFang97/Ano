import json
import string
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def one_hot(seq_batch,depth):

    # seq_batch.size() should be [seq,batch] or [batch,]
    # return size() would be [seq,batch,depth] or [batch,depth]
    #out = torch.zeros(seq_batch.size()+torch.Size([depth]))
    #dim = len(seq_batch.size())
    #index = seq_batch.view(seq_batch.size()+torch.Size([1]))
    #return out.scatter_(dim,index,1)

    dim = len(seq_batch.size()) 
    inp_ = torch.unsqueeze(seq_batch, dim)
    print(inp_)
    onehot = torch.zeros(seq_batch.size()+torch.Size([depth]))
    onehot.scatter_(dim, inp_, 1)

    return onehot

if __name__ == '__main__':
    d2_np = np.random.randint(5,size=(4,6))
    d1_np = np.random.randint(5,size=6)

    d1 = torch.from_numpy(d1_np)
    d2 = torch.from_numpy(d2_np)
    onehot1 = one_hot(d1, 5)
    onehot2 = one_hot(d2, 5)
    print("d1:\n", d1)
    print("o1:\n", onehot1)
    print("d1:\n", d2)
    print("o1:\n", onehot2)