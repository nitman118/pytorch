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


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = V, embedding_dim = 100)
        self.output = nn.Linear(in_features=100, out_features=V)

    def forward(self, t):
        #input
        t=t.reshape(-1,1)
        #embedding
        t = self.embedding(t)
        t = t.squeeze(1)
        t = F.log_softmax(self.output(t), dim=1)

        #done
        return t


model = Network()
optimizer = optim.SGD(model.parameters(),lr=0.025)
criterion = nn.NLLLoss()

dataset = TextDataset(x,y)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size = 32, shuffle=True, num_workers=2)

data_iter = iter(dataloader)
data = next(data_iter)
ws, cs =  data

print(cs.shape)
# out = model(ws)
# print(out)

model.train()


for epoch in range(5):
    loss_stat=0
    for data in dataloader:
        train_x, train_y = data
        preds = model(train_x)
        targets = preds.max(dim=1)[1]
        loss = criterion(preds,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_stat += loss.item()

    print(f'Epoch:{epoch}, loss:{loss_stat}')




