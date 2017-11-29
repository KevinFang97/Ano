import json
import string
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.cluster import KMeans
from model import *

use_cuda = torch.cuda.is_available()

#basic settings
use_SOS_token = False
EOS_token = 0
UNK_token = 1
SOS_token = 2
MAX_SEQ_LEN = 20 #use data_preparing to count max length

def one_hot(seq_batch,depth):

    # seq_batch.size() should be [seq,batch] or [batch,]
    # return size() would be [seq,batch,depth] or [batch,depth]
    #out = torch.zeros(seq_batch.size()+torch.Size([depth]))
    #dim = len(seq_batch.size())
    #index = seq_batch.view(seq_batch.size()+torch.Size([1]))
    #return out.scatter_(dim,index,1)

    dim = len(seq_batch.size()) 
    inp_ = torch.unsqueeze(seq_batch, dim)
    ##print(inp_)
    onehot = torch.zeros(seq_batch.size()+torch.Size([depth]))
    onehot.scatter_(dim, inp_, 1)

    return onehot

def translate_word(word,dict,n_words):
    try:
        if int(test_dict[word]) > n_words:
            w = "UNK"
        else:
            w = word
    except:
            w = "UNK"
    return w

def translate(words, dict, n_words):
    res = []
    for word in words:
        w = translate_word(word,dict,n_words)
        res.append(int(dict[w]))
    return res


def translate_back(ints, dict):
    res = []
    for i in ints:
        res.append(dict[i])
    return res

#translate qa into int sequence, add EOS padding
#return q, a, q_prob, a_prob, max_q_length, max_a_length, voca_size
def data_preparing(dict_path, prob_dict_path, qa_path, n_words):
    dict = json.loads(open(dict_path, "r").readline())
    reverse_dict = dict(zip(test_dict.values(), test_dict.keys()))
    prob_dict = json.loads(open(prob_dict_path, "r").readline())

    qa_file = open(qa_path, "r")
    ans = []
    ans_prob = []
    q = []
    q_prob = []
    qa_samples = []
    max_ans_length = 0
    max_q_length = 0
    qa_size = 0 #total q-a pairs
    debugging_size = 100 #for debugging, don't load all data
    debugging = True

    #scan qa_file
    print("scanning qa_file...")
    for line in qa_file:
        qa_samples.append(line)

        #detach q&a
        qa_samples[-1] = qa_samples[-1].translate(string.maketrans("", ""), string.punctuation)
        qa_samples[-1] = qa_samples[-1].strip("\r\n").split("\t")
        qa_samples[-1][0] = qa_samples[-1][0].split(" ") #q
        qa_samples[-1][1] = qa_samples[-1][1].split(" ") #a

        max_ans_length = max(max_ans_length, len(qa_samples[-1][1]))
        max_q_length = max(max_q_length, len(qa_samples[-1][0]))

        qa_size += 1

        #debugging use
        if debugging:
            if qa_size == debugging_size:
                qa_size = debugging_size
                break #for debugging
    print("qa file scanning complete, total ", qa_size, " pairs of qa scanned")

    #scan all qa pairs
    for i in range(qa_size):
        ans.append([])
        ans_prob.append([])
        q.append([])
        q_prob.append([])
        qa_pair_i = qa_samples[i]
        a_len = 0
        q_len = 0

        #for ans
        for word in qa_pair_i[1]:
            w = translate_word(w,dict,n_words)
            ans[-1].append(int(dict[w]))
            ans_prob[-1].append(float(prob_dict[w]))
            a_len += 1

        #for q
        for word in qa_pair_i[0]:
            w = translate_word(w,dict,n_words)
            q[-1].append(int(dict[w]))
            q_prob[-1].append(float(prob_dict[w]))
            q_len += 1

        #padding ans
        if a_len < max_ans_length:
            for j in range(j, max_ans_length):
                ans[-1].append(int(dict["<EOS>"]))
                ans_prob[-1].append(float(prob_dict["<EOS>"]))

        #padding q
        if q_len < max_q_length:
            for _ in range(k, max_q_length):
                q[-1].append(int(dict["<EOS>"]))
                q_prob[-1].append(float(prob_dict["<EOS>"]))

        if i%1000 + 1 == 0:
            print((i/1000+1),"k pairs of qa prepared")

    return qa_size, q, ans, q_prob, ans_prob, max_q_length, max_ans_length

#batch_q: [batch_size, max_q_length,] (torch var)
#batch_ans: [batch_size, max_ans_length,] (torch tensor)
#batch_topic: [batch_size,] (torch tensor)
#topic_size: int 
#voca_size: int
#topic_loss_weight: double
#word_loss_weight: double
def train_once(embedder, encoder, topic_picker, first_word_picker, decoder, batch_q, batch_ans, batch_topic, topic_size, voca_size, topic_loss_weight, word_loss_weight, embedder_optimizer, encoder_optimizer, topic_picker_optimizer, first_word_picker_optimizer, decoder_optimizer):
    batch_size = batch_q.size()[0]
    emb_q = embedder(batch_q) #batch__emb_q: [batch_size, max_q_length, embedded_size]
    
    #net1
    _, h = encoder(emb_q) #h: [batch_sz, 2*hid_sz]
    #test
    ##print("h:\n", h)

    #net2
    topic_score = topic_picker(h) #topic_score: [batch_sz, score_size]
    ##print("topic_score:\n", topic_score)

    #net3
    #pred_a: [batch_sz, max_ans_length, voca_size]
    one_hot_topic = one_hot(batch_topic, topic_size) #one_hot_topic: [batch_size, topic_size]
    one_hot_topic = Variable(one_hot_topic.float())
    batch_topic = Variable(batch_topic)
    one_hot_ans = one_hot(batch_ans, voca_size)
    one_hot_ans = Variable(one_hot_ans.float())
    batch_ans = Variable(batch_ans)
    
    ##print(h)
    ##print(one_hot_topic)
    ##print(one_hot_ans)
    pred_a = decoder(h, one_hot_topic, training=True, target=one_hot_ans)

    #net4
    #first_word_score: [batch_sz, voca_size]
    first_word_score = first_word_picker(h)

    #3 losses
    nll_loss = nn.NLLLoss()
    CE_loss = torch.nn.CrossEntropyLoss()
    #net2
    #print(topic_score)
    #print(batch_topic)
    topic_loss = CE_loss(topic_score, batch_topic)
    #net4
    #one_hot_first_word = one_hot(batch_ans[:][0], voca_size) #one_hot_first_word: [batch_size, voca_size]
    #first_words = batch_ans[:][0]
    first_words = batch_ans.narrow(1,0,1).squeeze(1)
    #print(first_word_score)
    #print(first_words)
    first_word_loss = nll_loss(first_word_score, first_words)
    #net3
    pred_a = pred_a.view(-1,voca_size)
    batch_ans = batch_ans.view(-1)
    #print(pred_a)
    #print(batch_ans)
    decoder_loss = CE_loss(pred_a, batch_ans)

    #total loss
    total_loss = topic_loss*topic_loss_weight + first_word_loss*word_loss_weight + decoder_loss
    total_loss /= batch_size
    
    total_loss.backward() 

    embedder_optimizer.step()
    encoder_optimizer.step()
    topic_picker_optimizer.step()
    first_word_picker_optimizer.step()
    decoder_optimizer.step()

    return topic_loss, first_word_loss, decoder_loss, total_loss


def train(batch_size, n_iters, n_words, embedded_size, hidden_size, topic_picker_hidden_size, topic_size, word_picker_latent_size, word_picker_hidden_size, dict_path, prob_dict_path, qa_path):
    #basic settings
    n_words = 10000 #use for test
    n_iters = 10000 #use for test
    batch_size = 16

    qa_size, q, ans, q_prob, ans_prob, max_q_length, max_ans_length = data_preparing(dict_path, prob_dict_path, qa_path, n_words)

    #torch variables init
    q = Variable(torch.LongTensor(q))
    if (use_cuda):
        q = q.cuda()

    #layers init (to be moved to main)
    embedder = nn.Embedding(n_words, embedded_size, padding_idx=EOS_token)
    encoder = Encoder(embedded_size, hidden_size)
    topic_picker = Picker(batch_size, hidden_size, topic_picker_hidden_size, topic_size)
    first_word_picker == LatentPicker(batch_size, hidden_size, word_picker_latent_size, word_picker_hidden_size, n_words, training=True)
    decoder = Decoder(batch_size, hidden_size, max_ans_length, topic_size, n_words)
    #classifier = Classifier(rnn_size, num_class)

    #optimizer init (to be moved to main)
    optim_vars = list(encoder.parameters()) + list(classifier.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(optim_vars, lr=1e-3)

    #validation sets init
    val_question = ["who", "are", "you", "<EOS>"]
    val_question_int = Variable(torch.LongTensor([translate(val_question, test_dict, n_words)]))
    embedded_val = embedder(val_question_int)
    if use_cuda:
        embedded_val = embedded_val.cuda()

    for iter in range(n_iters):
        #random select batch
        batch_index = np.random.randint(qa_size, size=batch_size)
        batch_q = q[batch_index]
        batch_q_prob = q_prob[batch_index]
        batch_ans = ans[batch_index]
        batch_ans_prob = ans_prob[batch_index]

        loss = train_once(batch_q, batch_q_prob, batch_ans, batch_ans_prob) #to be modified
        if (iter%100) == 99:
            print("Iteration: ", (iter + 1), "/", n_iters)
            print(loss)

'''
if __name__ == '__main__':
    batch_size = 16
    n_iters = 10000
    n_words = 10000
    embedded_size = 64
    hidden_size = 256 
    topic_picker_hidden_size = 256
    topic_size = 128
    word_picker_latent_size = 64
    word_picker_hidden_size = 256
    dict_path = ""
    prob_dict_path = ""
    qa_path = ""

    train(batch_size, n_iters, n_words, embedded_size, hidden_size, topic_picker_hidden_size, topic_size, word_picker_latent_size, word_picker_hidden_size, dict_path, prob_dict_path, qa_path)
'''