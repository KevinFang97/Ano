class UbuntuDataset(data.Dataset):
    def __init__(self, filepath, length_path):
        length_file = open(length_path, 'r')
        self.qa_size = int(length_file.readline())
        self.max_q_length = int(length_file.readline())
        self.max_ans_length = int(length_file.readline())
        length_file.close()
        
        print("loading data...")
        q = []
        ans = []
        q_prob = []
        ans_prob = []
        label = []
        count = 0
        while count < qa_size:
            
            count += 1
        print("{} entries".format(self.data_len))



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