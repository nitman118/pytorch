import torch
import  torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


requires_training = True


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
window_size=2
x, y = generate_cbow(corpus_cbow, 2, V)



if requires_training:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device:{device}')

    # read the file
    

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



    class CBOW(nn.Module):

        def __init__(self, dim=50):
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings=V, embedding_dim=dim)
            self.output = nn.Linear(in_features=dim, out_features=V)


        def forward(self, t):
            t = t.reshape(-1,4)#Shape: [32,4]
            t = self.embedding(t)#Shape:[32,4,50]
            t = t.mean(1)#Shape:[32,50], mean across words, 0:batch, 1:words, 
            t = self.output(t)#shape[32,V]
            return t

    dims = [50,150,300]

    for dim in dims:
        model = CBOW(dim).to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        dataset = TextDataset(x,y)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size = 32, shuffle=True, num_workers=2)

        # data_iter = iter(dataloader)
        # data = next(data_iter)
        # ws, cs =  data

        # print(ws.shape)
        # print(cs.shape)
        # out = model(ws)
        # print(out)

        model.train()

        for epoch in range(10):
            loss_stat=[]
            for data in dataloader:
                optimizer.zero_grad()

                train_x, train_y = data[0].to(device), data[1].to(device)

                preds = model(train_x)

                targets = train_y.max(dim=1)[1]
                # print(targets)
                loss = criterion(preds,targets)
                
                loss.backward()

                optimizer.step()

                loss_stat.append(loss.item())

            print(f'Epoch:{epoch}, Mean loss:{np.mean(loss_stat)}')


        model.eval()
        model.cpu()
        #
        # # TODO: Check this
        # embedding = model.embedding.weight.data.numpy()


        for name, param in model.named_parameters():
            print(name, param.shape)


        embedding = model.embedding.weight.detach().numpy()

        np.save(f'cbow_embed_{dim}.npy', embedding)


    

# Retrieving saved embeddings
# SG_50 = np.load('sem50.npy')
# SG_150 = np.load('sem150.npy')
# SG_300 = np.load('sem300.npy')
 
CBOW_50 = np.load('cbow_embed_50.npy')
CBOW_150 = np.load('cbow_embed_150.npy')
CBOW_300 = np.load('cbow_embed_300.npy')
embeddings = [CBOW_50,CBOW_150,CBOW_300]
embed_names = ['CBOW_50','CBOW_150','CBOW_300']

 #Embed a word by getting the one hot encoding and taking the dot product of this vector with the embedding matrix
# 'word' = string type
def embed(word, embedding, vocab_size = V, tokenizer=tokenizer):
    # get the index of the word from the tokenizer, i.e. convert the string to it's corresponding integer in the vocabulary
    int_word = tokenizer.texts_to_sequences([word])[0]
    # get the one-hot encoding of the word
    bin_word = to_categorical(int_word, V)
    return np.dot(bin_word, embedding)

# Display setting of the dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
 
#Input analogies
analogies=[('mouse','mice','lobster','lobsters','Plural nouns'),
           ('girl','boy','queen','king','Woman-man'),
           ('look','looking','watch','watching','Present participle'),
           ('is','are','was','were','Plural verbs'),
           ('sit','stand','stay','go','Opposite'),
           ('happy','sad','joys','sorrows','Opposite'),
           ('say','said','think','thought','Past tense'),
           ('is','was','are','were','Past tense'),
           ('small','smaller','large','larger','Comparative')]
 
# (we discard the input question words during this search)
 
df = pd.DataFrame(columns = ['Relationship','Analogy task','True word','sim1','Predicted word','sim2','Embedding','Correct?'])
for ipvector in analogies:
    for em, em_name in zip(embeddings, embed_names):
        prediction = embed(ipvector[1],em) - embed(ipvector[0],em) + embed(ipvector[2],em)
        true_word = ipvector[3]
        sim1 = cosine_similarity(embed(true_word, em), prediction)[0][0]
        # Discarding the input question words
        list_ipvector = [ipvector[0],ipvector[1],ipvector[2]] # to remove
        tokenizer2 = {a:tokenizer.word_index[a] for a in tokenizer.word_index.keys() if a not in list_ipvector}
        all_words = embed(list(tokenizer2.keys()), em)
        sim2_allwords = cosine_similarity(all_words, prediction)
        word_id = np.argmax(sim2_allwords)
        predicted_word = list(tokenizer2.keys())[word_id]
        sim2 = sim2_allwords[word_id][0]
        iscorrect = true_word==predicted_word
        df = df.append({'Relationship':ipvector[4],'Analogy task':f' \'{ipvector[0]}\' is to \'{ipvector[1]}\' as \'{ipvector[2]}\' is to ?','True word': true_word,
                       'sim1':sim1, 'Predicted word': predicted_word, 'sim2': sim2,'Embedding':em_name,'Correct?':iscorrect}, ignore_index=True)
Filt = df['Correct?']==True
print(df[Filt])
print(df)

