    import torch
    import  torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical

    import numpy as np



    # read the file
    file_name = '/home/nitman118/Documents/code/pytorch/WordEmbedding/alice.txt'
    corpus = open(file_name).readlines()
    #convert text into sentences
    corpus = [sentence for sentence in corpus if sentence.count(" ") >= 2]
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'+"'")

    tokenizer.fit_on_texts(corpus)

    corpus = tokenizer.texts_to_sequences(corpus)
    print(corpus[:5])
    n_samples = sum(len(s) for s in corpus) # total number of words in the corpus
    print(n_samples)
    V = len(tokenizer.word_index) + 1 # total number of unique words in the corpus + 1
    print(V)

    #generate data for Skipgram
    def generate_data_skipgram(corpus, window_size, V):
        maxlen = window_size*2
        all_in = []
        all_out = []
        for words in corpus:
            L = len(words)
            for index, word in enumerate(words):
                p = index - window_size
                n = index + window_size + 1

                in_words = []
                labels = []
                for i in range(p, n):
                    if i != index and 0 <= i < L:
                        # Add the input word
                        #in_words.append(word)
                        all_in.append(word)
                        # Add one-hot of the context words
                        all_out.append(to_categorical(words[i], V))

        return (np.array(all_in),np.array(all_out))


    corpus_cbow = []

    for sent in corpus:
        corpus_cbow.append([0,0]+sent+[0,0])


    # prepare data for CBOW
    def generate_cbow(corpus, window_size, V):
        maxlen = window_size*2 + 1
        const_len = []
        all_in = []
        all_out = []
        for sentence in corpus:
            L = len(sentence)
            for start_ind in range(L-maxlen+1):
                sent_part = sentence[start_ind:start_ind+maxlen]
                target = sent_part[window_size]
                sent_part.remove(sent_part[window_size])
                all_in.append(sent_part)
                all_out.append(to_categorical(target, V))

        return (np.array(all_in),np.array(all_out))

    # create training data

    x, y = generate_cbow(corpus_cbow, 2, V)
    window_size=2
    x, y = generate_data_skipgram(corpus, window_size, V)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, x, y):
            #data loading
            self.x = torch.from_numpy(x)
            self.y = torch.from_numpy(y)
            self.n_samples = x.shape[0]

        def __getitem__(self, index):
            return self.x[index], self.y[index]

        def __len__(self):
            return self.n_samples


    print(x.shape)
    print(y.shape)


    class Skipgram(nn.Module):

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings = V, embedding_dim = 50)
            self.output = nn.Linear(in_features=50, out_features=V)

        def forward(self, t):
            #input
            t=t.reshape(-1,1)
            #embedding
            t = self.embedding(t)
            t = t.squeeze(1)
            t = self.output(t)

            #done
            return t

    #Skipgram
    model = Skipgram() #model
    print(model)
    optimizer = optim.Adam(model.parameters(),lr=0.025)
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    dataset = TextDataset(x,y) #Dataset
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size = 32, shuffle=True, num_workers=2)

    # data_iter = iter(dataloader)
    # data = next(data_iter)
    # ws, cs =  data

    # print(cs.shape)
    # out = model(ws)
    # print(out)

    model.train()


    for epoch in range(10):
        loss_stat=[]
        for data in dataloader:
            train_x, train_y = data
            preds = model(train_x)
            targets = train_y.max(dim=1)[1]
            loss = criterion(preds,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_stat.append(loss.item())

        print(f'Epoch:{epoch}, loss:{np.mean(loss_stat)}')

    model.eval()




    #
    # # TODO: Check this
    # embedding = model.embedding.weight.data.numpy()
    #
    # # 'word' = string type
    # def embed(word, embedding=embedding, vocab_size = V, tokenizer=tokenizer):
    #     # get the index of the word from the tokenizer, i.e. convert the string to it's corresponding integer in the vocabulary
    #     int_word = tokenizer.texts_to_sequences([word])[0]
    #     # get the one-hot encoding of the word
    #     bin_word = to_categorical(int_word, V)
    #     return np.dot(bin_word, embedding)
    #
    # # TODO: compute the embedding by implementing the formula above. Use the embed function to get an embedding of a word
    # e_king = embed('king')
    # e_queen = embed('queen')
    # e_man = embed('man')
    # e_woman = embed('woman')
    # e_orange = embed('orange')
    # predictedEmbedding = (e_king - e_queen) + e_woman # expected embedding = e_man
    # print(f'Dist to man:{np.linalg.norm(e_man - predictedEmbedding, ord=2)}')
    #
    # min_dist = 100000
    # for word in tokenizer.word_index.keys():
    #   e_word = embed(word)
    #   new_dist = np.linalg.norm(e_word - predictedEmbedding, ord=2)
    #   if new_dist <min_dist:
    #     min_dist = new_dist
    #     new_word = word
    #
    # print(new_word)
    # print(min_dist)
    #
    # words = [i[0] for i in list((tokenizer.word_index.items()))]
    #
    # results = {}
    # for word in words:
    #     results[word] = np.linalg.norm(predictedEmbedding - embed(word))
    #
    # import pandas as pd
    # dfResults = pd.DataFrame.from_dict(results, orient='index').sort_values(0)
    # dfResults['index'] = range(1, len(dfResults)+1)
    # print(dfResults.iloc[:100, :])
    # print(dfResults.loc['man', :])
    # dfResults.to_excel('dist300.xlsx')
