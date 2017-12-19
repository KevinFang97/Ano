import json
import string
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.cluster import KMeans

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



def NoobDataset(dict_path, prob_dict_path, qa_path, n_words):
	qa_size, q, ans, q_prob, ans_prob, max_q_length, max_ans_length = data_preparing(dict_path, prob_dict_path, qa_path, n_words)
	