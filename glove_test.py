import numpy as np
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    line_count = 0;
    max_line = 10000 #for test
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding

        line_count += 1
        if line_count%1000 == 0:
            print("Total lines read = %dk" % (line_count/1000))
        if line_count > max_line:
            break

    print("Done.",len(model)," words loaded!")
    return model

#if __name__ == "__main__":
#    gloveFile = 'glove.6B/glove.6B.100d.txt'
#    embedder = loadGloveModel(gloveFile)