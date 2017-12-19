#test sample 
import json
import string
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.cluster import KMeans
from model import *
from main import *

def load_dict(filename):
    return json.loads(open(filename, "r").readline())

if __name__ == '__main__':
    n_words = 50
    embedded_size = 32
    hidden_size = 16
    batch_size = 4
    topic_picker_hidden_size = 8
    topic_size = 4
    word_picker_latent_size = 4
    word_picker_hidden_size = 8
    voca_size = n_words
    max_ans_length = 10
    max_q_length = 15
    learning_rate = 0.01

    input_dir='./test/data/'
    vocab = load_dict(input_dir+'vocab.json')

    np_q = np.random.randint(n_words, size=(1,max_q_length))

#batch_q: [batch_size, max_q_length,] (torch var)
#batch_ans: [batch_size, max_a_length,] (torch var)
#batch_topic: [batch_size,] (torch var)
    batch_q = torch.from_numpy(np_q )

    embedder = nn.Embedding(n_words, embedded_size, padding_idx=EOS_token)
    encoder = Encoder(embedded_size, hidden_size)
    topic_picker = Picker(batch_size, hidden_size, topic_picker_hidden_size, topic_size)
    first_word_picker = LatentPicker(batch_size, hidden_size, word_picker_latent_size, word_picker_hidden_size, voca_size, training=True)
    decoder = Decoder(batch_size, hidden_size, max_ans_length, topic_size, voca_size)

    
    print(sample(embedder, encoder, topic_picker, first_word_picker, decoder, batch_q, vocab))