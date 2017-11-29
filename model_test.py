import model
import glove_test
import pre_model_test
import numpy as np
from torch.autograd import Variable

import torch

def encoder_test():
    embedded_sentence = pre_model_test.embedder_test()
    embedded_sentence = Variable(torch.FloatTensor(embedded_sentence))
    encoder = model.Encoder(100,50)
    _, h = encoder(embedded_sentence)
    print(h.size())
    return h

def picker_test(h):
    picker = model.Picker(3,50,50,10)
    topic_score = picker(h)
    print("topic_score: ")
    print(topic_score)
    return topic_score

def latent_picker_test(h):
    lpicker = model.LatentPicker(3,50,10,50,20)
    topic_score = lpicker(h)
    print("word_score: ")
    print(topic_score)
    return topic_score

def decoder_test(h,score,l_score):
    decoder = model.Decoder(3,50,6,10,20)
    target = Variable( torch.from_numpy( np.random.rand(3,6,20) ).float() )
    first_word = Variable( torch.from_numpy( np.random.rand(3,1,20)/10 ).float() )
    training_out = decoder(h,score,training=True,target=target)
    pred_out = decoder(h,score,training=False,first_word=first_word)
    print("training_out:")
    print(training_out)
    print("pred_out:")
    print(pred_out)


if __name__ == "__main__":
    h = encoder_test()
    score = picker_test(h)
    l_score = latent_picker_test(h)
    print("h_size: ", h.size())
    decoder_test(h,score,l_score)