import json
import string
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.cluster import KMeans


from helper import showPlot, timeSince, sent2indexes, indexes2sent
from model import *

#use_cuda = torch.cuda.is_available()
use_cuda = False

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

#batch_q: [batch_size, max_q_length,] (torch var)
#batch_ans: [batch_size, max_ans_length,] (torch tensor)
#batch_topic: [batch_size,] (torch tensor)
#topic_size: int 
#voca_size: int
#topic_loss_weight: double
#word_loss_weight: double
def train_once(embedder, encoder, topic_picker, first_word_picker, decoder, 
	batch_q, batch_ans, batch_topic, topic_size, voca_size, topic_loss_weight, word_loss_weight, 
	embedder_optimizer, encoder_optimizer, topic_picker_optimizer, first_word_picker_optimizer, decoder_optimizer, nll_loss, CE_loss):
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

#need support:
#1. data loader
def train(embedder, encoder, topic_picker, first_word_picker, decoder, 
        learning_rate, data_loader, topic_size, voca_size, 
        save_every, sample_every, print_every, plot_every,
        model_dir, vocab):
    start = time.time()
    print_time_start = start
    plot_losses = []
    print_loss_total, print_loss_topic, print_loss_word, print_loss_decoder = 0., 0., 0., 0.

    embedder_optimizer = optim.Adam(embedder.parameters(), lr = learning_rate)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
    topic_picker_optimizer = optim.Adam(topic_picker.parameters(), lr = learning_rate)
    first_word_picker_optimizer = optim.Adam(first_word_picker.parameters(), lr = learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)


    nll_loss = nn.NLLLoss()
    CE_loss = torch.nn.CrossEntropyLoss()

    data_iter = iter(data_loader)

    for it in range(1, n_iters + 1):
        batch_q, batch_ans, batch_topic = data_iter.next()

        #anneal weight?
        topic_loss_weight = 0.2
        word_loss_weight = 0.2

        topic_loss,first_word_loss,decoder_loss,total_loss = train_once(embedder, encoder, topic_picker, first_word_picker, decoder, 
                batch_q, batch_ans, batch_topic, topic_size, voca_size, topic_loss_weight, word_loss_weight, 
                embedder_optimizer, encoder_optimizer, topic_picker_optimizer, first_word_picker_optimizer, decoder_optimizer, 
                nll_loss, CE_loss)

        print_loss_total += total_loss
        print_loss_decoder += decoder_loss
        print_loss_word += first_word_loss
        print_loss_topic += topic_loss

        if it % save_every ==0:
            if not os.path.exists('%sthree_net_%s/' % (model_dir, str(it))):
                os.makedirs('%sthree_net_%s/' % (model_dir, str(it)))
            torch.save(f='%sthree_net_%s/embedder.pckl' % (model_dir, str(it)),obj=embedder)
            torch.save(f='%sthree_net_%s/encoder.pckl' % (model_dir,str(it)),obj=encoder)
            torch.save(f='%sthree_net_%s/topic_picker.pckl' % (model_dir,str(it)),obj=topic_picker)
            torch.save(f='%sthree_net_%s/first_word_picker.pckl' % (model_dir,str(it)),obj=first_word_picker)
            torch.save(f='%sthree_net_%s/decoder.pckl' % (model_dir,str(it)),obj=decoder)
        if it % sample_every == 0:
            samp_idx=np.random.choice(len(batch_q),4) #pick 4 samples
            for i in samp_idx:
                question, target = batch_q[i].view(1,-1), batch_ans[i].view(1,-1)
                sampled_sentence = sample(embedder, encoder, topic_picker, first_word_picker, decoder, question, vocab)
                ivocab = {v: k for k, v in vocab.items()}
                print('question: %s'%(indexes2sent(question.squeeze().numpy(), ivocab, ignore_tok=EOS_token)))
                print('target: %s'%(indexes2sent(target.squeeze().numpy(), ivocab, ignore_tok=EOS_token)))
                print('predicted: %s'%(sampled_sentence))
        #print and plot
        if it % print_every == 0:
            print_loss_total = print_loss_total / print_every
            print_loss_word = print_loss_word / print_every
            print_loss_topic = print_loss_topic / print_every
            print_loss_decoder = print_loss_decoder / print_every
            print_time=time.time()-print_time_start
            print_time_start=time.time()
            print('iter %d/%d  step_time:%ds  total_time:%s total_loss: %.4f topic_loss: %.4f first_word_loss: %.4f dec_loss: %.4f'%(it, n_iters, 
                  print_time, timeSince(start, it/n_iters), print_loss_total, print_loss_topic, print_loss_word, print_loss_decoder))
        
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            print_loss_total, print_loss_topic, print_loss_word, print_loss_decoder = 0., 0., 0., 0.
    
    showPlot(plot_losses)

def sample(embedder, encoder, topic_picker, first_word_picker, decoder, question, vocab):
    ivocab = {v: k for k, v in vocab.items()}
    q = Variable(question) # shape: [batch_sz (=1) x seq_len]
    q = q.cuda() if use_cuda else q
    
    #emb
    emb_q = embedder(q)

    #net1
    _, h = encoder(emb_q)

    #net2
    topic_score = topic_picker(h)
    max_topic_score = torch.max(topic_score)
    topic_score = (topic_score >= max_topic_score).float()

    #net4
    first_word_score = first_word_picker(h)
    max_word_score, _ = torch.max(first_word_score, 1)
    first_word_score = (first_word_score >= max_word_score).float()

    #net3
    pred = decoder(h, topic_score, training=False, first_word=first_word_score)
    pred = torch.squeeze(pred, 0)

    decoded_words = []
    seq_size = pred.shape[0]
    for i in range(seq_size):
        _, decoder_output = torch.max(pred[i], 0)
        word_index = decoder_output.data.cpu().numpy()[0]
        #print(i, ": ", word_index)
        if word_index == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(ivocab[word_index])

    return ' '.join(decoded_words)




'''
def train(batch_size, n_iters, n_words, embedded_size, hidden_size, topic_picker_hidden_size, topic_size, word_picker_latent_size, word_picker_hidden_size, dict_path, prob_dict_path, qa_path):
    #basic settings
    n_words = 100 #use for test
    n_iters = 100 #use for test
    batch_size = 16
    embedded_size = 32
    hidden_size = 16
    topic_picker_hidden_size = 8
    topic_size = 4
    word_picker_latent_size = 4
    word_picker_hidden_size = 8
    voca_size = n_words
    max_ans_length = 10
    max_q_length = 15
    learning_rate = 0.01

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