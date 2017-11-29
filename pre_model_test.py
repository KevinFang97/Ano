import pre_model
import glove_test
import numpy as np

def embedder_test():
    f = open('pre_model_test.txt','r')
    #init sentence_batch
    sentence_batch = []
    for line in f:
        splitLine = line.split()
        sentence_batch.append(splitLine)
    #init max_size
    max_size = 10
    #init glove_embedded_size
    glove_embedded_size = 100
    #init glove_dict
    gloveFile = 'glove.6B/glove.6B.100d.txt'
    glove_dict = glove_test.loadGloveModel(gloveFile)
    #init UNK_vec
    UNK_vec = np.ones((glove_embedded_size,))
    #init EOS_vec
    EOS_vec = np.zeros((glove_embedded_size,))

    #test
    embedded_sentence = pre_model.embedder(sentence_batch,max_size,glove_embedded_size,glove_dict,UNK_vec,EOS_vec)

    #print
    count = 0
    for sentence in embedded_sentence:
        count += 1
        print("The ", count, "th sentence embedded: ")
        print(sentence)
    print("Done! ", count, " sentences embedded in total.")

    return embedded_sentence


def classifier_test(embedded_sentence):
	class_label = pre_model.classifier(embedded_sentence,None)
	print(class_label)


#if __name__ == "__main__":
#	embedded_sentence = embedder_test()
#	classifier_test(embedded_sentence)