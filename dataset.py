import numpy as np

#store all data in memory:
#use 3 numpy arr: q, ans, topic

#retrieved these data from .txt --- written in preprocess.py

class UbuntuDataset(data.Dataset):
    def __init__(self, filepath, length_path, dataset_path):
        #length_path = "data/data_length.txt"
        #dataset_path = "data/train_dataset.txt" #or valid, test

        length_file = open(length_path,'r')
        dataset_file = open(dataset_path, 'r')

        #read length
        _ = length_file.readline()
        self.max_q_length = int(length_file.readline())
        self.max_ans_length = int(length_file.readline())
        length_file.close()

        #read dataset size
        self.qa_size = int(dataset_file.readline())
        print("loading data from ", dataset_path)
        print("qa_size = ", self.qa_size)
        #init 3 lists
        q = np.zeros((qa_size, max_q_length), dtype='int32')
        ans = np.zeros((qa_size, max_ans_length), dtype='int32')
        label = np.zeros((qa_size,), dtype='int16')
        count = 0
        #read data
        while count < qa_size:
            #implicit str->int
            q[count] = dataset_file.readline().split()
            ans[count] = dataset_file.readline().split()
            _ = dataset_file.readline()
            _ = dataset_file.readline()
            label[count] = dataset_file.readline()
            count += 1
        print("{} entries".format(self.qa_size))


    #TODO !!! 
    #(may need changes in train_once so that np -> torch happen there)
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