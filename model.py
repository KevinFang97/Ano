from math import sqrt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

#net1
#BiGRU+concat+nonlinear
class Encoder(nn.Module):
    def __init__(self, emb_sz, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(emb_sz, hidden_size, dropout=0.2, batch_first=True, bidirectional=True)
        for w in self.gru.parameters(): # initialize the gate weights with orthogonal\
            if w.dim()>1:
                nn.init.orthogonal(w, sqrt(2))

    #embedded: [batch_sz, seq_len, emb_sz]
    #out: [batch_sz, seq_len, hid_sz*2] (biRNN)   hidden: [batch_sz, 2*hid_sz]
    def forward(self, embedded):
        output, hidden = self.gru(embedded) #hidden: [2, batch_sz, hid_sz] (2=fw&bw)
        N = list(hidden.size())[1]
        hidden = torch.transpose(hidden, 0, 1).contiguous() #hidden: [batch_sz, 2, hid_sz] (2=fw&bw)
        hidden = hidden.view(N,-1) #hidden: [batch_sz, 2*hid_sz] (2=fw&bw)
        hidden = torch.sinh(hidden) #hidden: [batch_sz, 2*hid_sz] (2=fw&bw)
        return output, hidden

#net2, net4
#used to pick topic and first word of answer
#dropout+dense+relu+dense+softmax
class Picker(nn.Module):
    def __init__(self, batch_sz, hidden_size, relu_size, output_size):
        super(Picker, self).__init__()
        self.relu = nn.PReLU()
        self.l1 = nn.Linear(hidden_size*2, relu_size)
        self.l2 = nn.Linear(relu_size, output_size)
        self.drop = nn.Dropout(p=0.2)
        self.sm = nn.Softmax()

    #input hidden: [batch_sz, 2*hidden_size]
    #output score: [batch_sz, score_size]
    def forward(self, hidden):
        hidden = self.drop(hidden)
        hidden = self.l1(hidden) #hidden: [batch_sz, relu_size]
        hidden = self.relu(hidden) #hidden: [batch_sz, relu_size]
        score = self.l2(hidden)
        score = self.sm(score)
        return score

#net4
#used to pick topic and first word of answer
#dropout+dense+relu+dense+softmax
class LatentPicker(nn.Module):
    def __init__(self, batch_sz, hidden_size, latent_size, relu_size, output_size, training=False):
        super(LatentPicker, self).__init__()
        self.relu = nn.PReLU()
        self.l_mu = nn.Linear(hidden_size*2, latent_size)
        self.l_logvar = nn.Linear(hidden_size*2, latent_size)
        self.l1 = nn.Linear(hidden_size*2 + latent_size, relu_size)
        self.l2 = nn.Linear(relu_size, output_size)
        self.sm = nn.Softmax()
        self.drop = nn.Dropout(p=0.2)
        self.training = training

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    #input hidden: [batch_sz, 2*hidden_size]
    #output score: [batch_sz, score_size]
    def forward(self, hidden):
        mu = self.l_mu(hidden)
        logvar = self.l_logvar(hidden)
        z = self.reparameterize(mu, logvar) #z: [batch_sz, latent_size]
        hidden = torch.cat( (hidden, z) , dim=1) #hidden: [batch_sz, 2*hidden_size+latent_size]
        hidden = self.drop(hidden) #hidden: [batch_sz, 2*hidden_size+latent_size]
        hidden = self.l1(hidden) #hidden: [batch_sz, relu_size]
        hidden = self.relu(hidden) #hidden: [batch_sz, relu_size]
        score = self.l2(hidden)
        score = self.sm(score)
        return score

#net3
#concat hidden and topic_score(softmaxed) as hidden_var
#use a1(a1_score when pred) as start
#use teach_force while is_training
class Decoder(nn.Module):
    def __init__(self, batch_sz, hidden_size, max_ans_length, topic_size, voca_size, use_cuda=torch.cuda.is_available()):
        super(Decoder, self).__init__()
        self.hidden_var_size = 2*hidden_size+topic_size
        self.voca_size = voca_size
        self.gru = nn.GRU(self.voca_size, self.hidden_var_size,  dropout=0.2, batch_first=True)
        self.max_ans_length = max_ans_length
        self.sm = nn.Softmax()
        self.relu = nn.PReLU()
        self.l = nn.Linear(self.hidden_var_size, voca_size)
        for w in self.gru.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                nn.init.orthogonal(w, sqrt(2))

    def step(self, input, hidden):
        hidden.detach_()
        output, hidden = self.gru(input, hidden)
        output = self.relu(self.l(output))
        return output, hidden

    #hidden: [batch_sz, 2*hidden_size]
    #topic_score: [batch_sz, topic_size]
    #target: [batch_sz, max_ans_length, voca_size]
    #first_word: [batch_sz, 1, voca_size]
    #output: [batch_sz, max_ans_length, voca_size]
    def forward(self, hidden, topic_score, training=False, target=None, first_word=None):
        #test
        ##print("########decoder test start########\n")
        ##print("h:\n", hidden)
        ##print("topic_score:\n", topic_score)
        ##print("########decoder test end^^########\n")
        #prepare hidden variable
        hidden_var = torch.cat((hidden, topic_score), 1) #hidden_var: [batch_sz, hidden_var_size]
        N,_= hidden_var.size()
        hidden_var = hidden_var.view(1,N,self.hidden_var_size) #hidden_var: [1, batch_sz, hidden_var_size]

        #prepare first_word
        if training:
            first_word = target.narrow(1,0,1) #shape: [batch_sz, 1, voca_size]

        #init
        output = first_word #shape: [batch_sz, 1, voca_size]
        new_word = first_word

        #loop
        for time_step in range(1, self.max_ans_length):
            #output: [batch_sz, time_step, voca_size]

            #prepare prev_word
            if training:
                prev_word = target.narrow(1,(time_step-1),1)
            else:
                prev_word = new_word
            ##print(prev_word)
            prev_word = self.relu(prev_word)

            #step
            new_word, hidden_var = self.step(prev_word, hidden_var) #new_word: [batch_sz x 1 x hid_sz] (1=seq_len)
            
            #softmax
            new_word = torch.squeeze(new_word, dim=1)
            new_word = self.sm(new_word)
            new_word = new_word.unsqueeze(1)

            #store in output
            output = torch.cat((output, new_word), 1) #output: [batch_sz, time_step + 1, voca_size]

        #out_put: [batch_sz, max_ans_length, voca_size]
        return output
