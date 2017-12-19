import string
import json
import collections

if __name__ == '__main__':

    # convert opensubtitle data into TFRecord, and save the dictionary (int -> word) to json file

    train_data_load_path = "data/train.txt"
    # int -> word dictionary
    dictionary_save_path = "data/dictionary.json"
    # how many words should be added into dictionary
    num_words = 9
    # larger batch size speeds up the process but needs larger memory
    process_batch_size = 2
    # (for validating) determine maximum question-answer pairs to convert & save, use -1 to process all pairs
    train_num_pairs = -1

    train_file = open(train_data_load_path, "r")
    dictionary_file = open(dictionary_save_path, "w")

    train_linecount = 0
    if train_num_pairs == -1:
        for train_linecount, _ in enumerate(train_file):
            pass
        train_linecount += 1
    else:
        train_linecount = train_num_pairs

    words = []

    #process and save QA pairs
    print("Creating dictionary...")
    train_file.seek(0)
    counter = collections.Counter()
    for i in range(train_linecount):
        line = train_file.readline().translate(
            string.maketrans("", ""), string.punctuation)
        if line == "":
            break
        words += line.split()
        # texts.append([question, answer])
        if i % process_batch_size == 0 and i:
            print(str(i) + "/" + str(train_linecount))
            counter.update(
                dict(collections.Counter(words).most_common(num_words)))
            words = []
    counter.update(dict(collections.Counter(words).most_common(num_words)))
    del words
    train_file.close()

    count = [['UNK', -1], ['<GO>', -1], ['<EOS>', -1]]
    count.extend(counter.most_common(num_words - 1))  # minus 1 for UNK
    del counter
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    dictionary_file.write(json.dumps(reversed_dictionary))
    del reversed_dictionary