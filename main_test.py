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


if __name__ == '__main__':
    n_words = 10
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

    np_q = np.random.randint(n_words, size=(batch_size, max_q_length))
    np_ans = np.random.randint(n_words, size=(batch_size, max_ans_length))
    np_topic = np.random.randint(topic_size, size=4)

#batch_q: [batch_size, max_q_length,] (torch var)
#batch_ans: [batch_size, max_a_length,] (torch var)
#batch_topic: [batch_size,] (torch var)
    batch_q = Variable( torch.from_numpy(np_q ))
    batch_ans = torch.from_numpy(np_ans )
    batch_topic = torch.from_numpy(np_topic )
    topic_loss_weight = 0.5
    word_loss_weight = 0.5


    embedder = nn.Embedding(n_words, embedded_size, padding_idx=EOS_token)
    encoder = Encoder(embedded_size, hidden_size)
    topic_picker = Picker(batch_size, hidden_size, topic_picker_hidden_size, topic_size)
    first_word_picker = LatentPicker(batch_size, hidden_size, word_picker_latent_size, word_picker_hidden_size, voca_size, training=True)
    decoder = Decoder(batch_size, hidden_size, max_ans_length, topic_size, voca_size)

    embedder_optimizer=optim.Adam(embedder.parameters(), lr=learning_rate)
    encoder_optimizer=optim.Adam(encoder.parameters(), lr=learning_rate)
    topic_picker_optimizer=optim.Adam(topic_picker.parameters(), lr=learning_rate)
    first_word_picker_optimizer=optim.Adam(first_word_picker.parameters(), lr=learning_rate)
    decoder_optimizer=optim.Adam(decoder.parameters(), lr=learning_rate)

    l1,l2,l3,l0 = train_once(embedder, encoder, topic_picker, first_word_picker, decoder, batch_q, batch_ans, batch_topic, topic_size, voca_size, topic_loss_weight, word_loss_weight, embedder_optimizer, encoder_optimizer, topic_picker_optimizer, first_word_picker_optimizer, decoder_optimizer)
    print(l1)
    print(l2)
    print(l3)
    print(l0)