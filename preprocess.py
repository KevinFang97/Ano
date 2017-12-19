import json
import string
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.cluster import KMeans

#preprocess procedure:
#1.create dict, prob_dict (done)
#2.translate q, a; get q_prob, a_prob (done)
#3.get cluster (TODO)
#4.write into file (done)

#5.create dataset from file (TODO)

def translate_word(word,dictionary,n_words):
    try:
        if int(dictionary[word]) > n_words:
            w = "UNK"
        else:
            w = word
    except:
            w = "UNK"
    return w

def translate(words, dictionary, n_words):
    res = []
    for word in words:
        w = translate_word(word,dictionary,n_words)
        res.append(int(dictionary[w]))
    return res


def translate_back(ints, dictionary):
    res = []
    for i in ints:
        res.append(dictionary[i])
    return res

#translate qa into int sequence, add EOS padding
#return q, a, q_prob, a_prob, max_q_length, max_a_length, voca_size
def data_preparing(dict_path, prob_dict_path, qa_path, n_words):
    dictionary = json.loads(open(dict_path, "r").readline())
    reverse_dict = dict(zip(dictionary.values(), dictionary.keys()))
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

        max_ans_length = max(max_ans_length, len(qa_samples[-1][1]) + 1)
        max_q_length = max(max_q_length, len(qa_samples[-1][0]) + 1)

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
            word = translate_word(word,dictionary,n_words)
            ans[-1].append(int(dictionary[word]))
            ans_prob[-1].append(float(prob_dict[word]))
            a_len += 1

        #for q
        for word in qa_pair_i[0]:
            word = translate_word(word,dictionary,n_words)
            q[-1].append(int(dictionary[word]))
            q_prob[-1].append(float(prob_dict[word]))
            q_len += 1

        #padding ans
        if a_len < max_ans_length:
            for j in range(a_len, max_ans_length):
                ans[-1].append(int(dictionary["<EOS>"]))
                ans_prob[-1].append(float(prob_dict["<EOS>"]))

        #padding q
        if q_len < max_q_length:
            for j in range(q_len, max_q_length):
                q[-1].append(int(dictionary["<EOS>"]))
                q_prob[-1].append(float(prob_dict["<EOS>"]))

        if i%1000 + 1 == 0:
            print((i/1000+1),"k pairs of qa prepared")

    return qa_size, q, ans, q_prob, ans_prob, max_q_length, max_ans_length




'''
class NoobDataset(data.Dataset):
    def __init__(self, ):

    def __getitem__(self, offset):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        pos, q_len, a_len =  self.index[offset]['pos'], self.index[offset]['q_len'], self.index[offset]['a_len']
        question=self.data[pos:pos + q_len].astype('int64')
        answer=self.data[pos+q_len:pos+q_len+a_len].astype('int64')
        
        ## Padding ##
        if len(question)<self.max_seq_len:
            question=np.append(question, [0]*self.max_seq_len)    
        question=question[:self.max_seq_len]
        question[-1]=0
        if len(answer)<self.max_seq_len:
            answer=np.append(answer,[0]*self.max_seq_len)
        answer=answer[:self.max_seq_len]
        answer[-1]=0
        
        ## get real seq len
        q_len=min(int(q_len),self.max_seq_len) # real length of question for training
        a_len=min(int(a_len),self.max_seq_len) 
        return question, answer, q_len, a_len

    def __len__(self):
return self.data_len
'''

#just for test
def answer_clustering(ans, ans_prob, max_ans_length):
    return np.zeros_like(ans)



if __name__ == '__main__':
    dict_path = "data/dictionary.json"
    prob_dict_path = "data/prob_dict.json"
    qa_path = "data/train.txt"
    n_words = 9

    length_path = "data/data_length.txt"
    train_dataset_path = "data/train_dataset.txt"
    valid_dataset_path = "data/train_dataset.txt"
    test_dataset_path = "data/train_dataset.txt"

    qa_size, q, ans, q_prob, ans_prob, max_q_length, max_ans_length = data_preparing(dict_path, prob_dict_path, qa_path, n_words)

    label = answer_clustering(ans, ans_prob, max_ans_length) #need to be implemented

    length_file = open(length_path,'w')
    train_file = open(train_dataset_path, 'w')
    valid_file = open(valid_dataset_path, 'w')
    test_file = open(test_dataset_path, 'w')

    train_size = int(qa_size*0.7)
    valid_size = int(qa_size*0.2)
    #else is test

    length_file.write(str(qa_size))
    length_file.write("\n")
    length_file.write(str(max_q_length))
    length_file.write("\n")
    length_file.write(str(max_ans_length))
    length_file.close()

    for i in range(train_size):
        train_file.write(' '.join(map(str, q[i])))
        train_file.write("\n")
        train_file.write(' '.join(map(str, ans[i])))
        train_file.write("\n")
        train_file.write(' '.join(map(str, q_prob[i])))
        train_file.write("\n")
        train_file.write(' '.join(map(str, ans_prob[i])))
        train_file.write("\n")
        train_file.write(' '.join(map(str, label[i])))
        train_file.write("\n")
    train_file.close()

    for i in range(train_size, train_size + valid_size):
        valid_file.write(' '.join(map(str, q[i])))
        valid_file.write("\n")
        valid_file.write(' '.join(map(str, ans[i])))
        valid_file.write("\n")
        valid_file.write(' '.join(map(str, q_prob[i])))
        valid_file.write("\n")
        valid_file.write(' '.join(map(str, ans_prob[i])))
        valid_file.write("\n")
        valid_file.write(' '.join(map(str, label[i])))
        valid_file.write("\n")
    valid_file.close()
    for i in range(valid_size, qa_size):
        test_file.write(' '.join(map(str, q[i])))
        test_file.write("\n")
        test_file.write(' '.join(map(str, ans[i])))
        test_file.write("\n")
        test_file.write(' '.join(map(str, q_prob[i])))
        test_file.write("\n")
        test_file.write(' '.join(map(str, ans_prob[i])))
        test_file.write("\n")
        test_file.write(' '.join(map(str, label[i])))
        test_file.write("\n")
    test_file.close()








