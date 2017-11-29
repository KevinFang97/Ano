import numpy as np

#use GLoVe
#imput string arr size [N, sentence_length_not_same]
#imput numpy arr size [N, max_size, glove_embedded_size]
def embedder(sentence_batch,max_size,glove_embedded_size,glove_dict,UNK_vec,EOS_vec):
    N = len(sentence_batch)
    embedded_sentences = np.zeros((N,max_size,glove_embedded_size))

    for sentence_index in range(N):
        word_count = 0

        #word translation
        for word_index in range(len(sentence_batch[sentence_index])):
            word = sentence_batch[sentence_index][word_index]
            if word in glove_dict:
                embedded_sentences[sentence_index][word_index] = glove_dict[word]
            else:
                embedded_sentences[sentence_index][word_index] = UNK_vec #not in dict then use UNK
            word_count += 1

        #error handling
        if word_count >= max_size:
            print("ERROR: word_count ", word_count, " >= max_size ", max_size)

        #padding with EOS
        embedded_sentences[sentence_index][(word_count+1):] = EOS_vec

    return embedded_sentences

#pretrained answer classification
#input: a batch of embedded sentence of same dict
def classifier(embedded_answers, useful_param_to_predict):
    N = embedded_answers.shape[0]
    class_label = np.zeros((N,)) #trivial version, TO BE IMPLEMENTED (unsupervised learning on result of wr+pca)
	
    return class_label