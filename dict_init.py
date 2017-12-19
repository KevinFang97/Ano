import sys
import torch 
import torch.utils.data as data
import tables
import numpy as np
import unicodedata
import re
import string
import json
import collections

def textproc(s):
    s=s.lower()
    s=s.replace('\'s ','is ')
    s=s.replace('\'re ','are ')
    s=s.replace('\'m ', 'am ')
    s=s.replace('\'ve ', 'have ')
    s=s.replace('\'ll ','will ')
    s=s.replace('n\'t ', 'not ')
    s=s.replace(' wo not',' will not')
    s=s.replace(' ca not',' can not')
    s=re.sub('[\!;-]+','',s)
    s=re.sub('\.+','.',s)
    if s.endswith(' .'):
        s=s[:-2]
    s=re.sub('\s+',' ',s)
    s=s.strip()
    return s

def create_dict(train_file, vocab_size):
    file=open(train_file,'r')
    counter = collections.Counter()
    line_count = 0.0
    word_count = 0.0
    for i, qaline in enumerate(file):
        line_count += 1
        line = qaline.translate(string.maketrans("", ""), string.punctuation)
        if line == "":
            break
        line=textproc(line)
        words = line.split()
        word_count += len(words)
        counter.update(words)
        if i % process_batch_size == 0 and i:
            print(str(i))
    file.close()

    dictionary = {'UNK': 2, '<SOS>':1, '<EOS>': 0}
    prob_dict = {'UNK': line_count/word_count, '<SOS>':line_count/word_count, '<EOS>': line_count/word_count}
    count=counter.most_common(vocab_size - 3)  # minus 1 for UNK
    for word, freq in count:
        if word=='':
            continue
        dictionary[word] = len(dictionary)
        prob_dict[word] = float(freq) / word_count

    return dictionary, prob_dict

if __name__ == '__main__':

    data_path="./data/"
    file_in = data_path+"train.txt"
    # int -> word dict
    dict_path = data_path+"dictionary.json"
    prob_dict_path = data_path+"prob_dict.json"
    # how many words should be added into dict
    vocab_size = 9
    # larger batch size speeds up the process but needs larger memory
    process_batch_size = 3
    
    print ("creating dictionary...")
    vocab, prob_dict =create_dict(file_in, vocab_size)
    dict_file = open(dict_path, "w")
    dict_file.write(json.dumps(vocab))
    prob_dict_file = open(prob_dict_path, "w")
    prob_dict_file.write(json.dumps(prob_dict))