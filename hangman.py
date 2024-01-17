from torch import nn, tensor
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from tqdm import tqdm
from random import sample, uniform
import math

# This method generates roughly 2.5 million datapoints.
def gen_data(filename):
    f = open(filename, 'r')
    g = open('dataset.txt', 'w')
    i = 0
    for line in tqdm(f, total=227300, desc="Generating data!"):
        word = line[:-1]
        #print(word)
        unique = list(set(word))
        # if <= 5 unique chars, then fewer blanks
        if len(unique) <= 5:
            m = min(4, len(unique))
            sets = [s for i in range(1, m+1) for s in combinations(unique, i)]

        # if >= 10, then more
        elif len(unique) >= 10:
            m = len(unique)-4
            sets = sets = [s for i in range(m, len(unique)+1) for s in combinations(unique, i)]
        
        # otherwise, intermediate
        else: 
            m = min(6, len(unique))
            sets = sets = [s for i in range(4, m+1) for s in combinations(unique, i)]

        # Control size
        if len(sets) > 500:
            sets = sample(sets, 500)

        for s in sets:
            w = list(word)
            for i in range(len(word)):
                if word[i] in s: w[i] = '_'
            #print(w)
            for c in s:
                #print(c)
                #print(word.count(c))
                # Add the letter as many times as it occurs in the word,
                # to introduce a bias towards more frequent letters.
                for _ in range(word.count(c)):
                    g.write(word + ' ')
                    g.write(''.join(w) + ' ')
                    g.write(c + '\n')
                    #self.words.append(w + [0]*(29 - len(w))) # 29 is max word length
                    #self.guesses.append(ord(c)-ord('a'))
    g.close()

DEVICE = torch.device('cuda:0')
class HangmanData(Dataset):
    def __init__(self):
        try:
            f = open('dataset.txt', 'r')
            print("Taking data from dataset.txt")
        except FileNotFoundError:
            gen_data('words_250000_train.txt')
            f = open('dataset.txt', 'r')
        
        self.words = []
        self.guesses = []
        for line in tqdm(f, total=262595417):
            if uniform(0, 1) > 0.1: continue
            whole, word, guess = line[:-1].split(' ')
            # 26 is blank; 27 is pad
            self.words.append([ord(c)-ord('a') if c != '_' else 26 for c in word] + [27]*(29-len(word)))
            self.guesses.append(ord(guess)-ord('a'))

        print("Tensorifying")
        self.words = tensor(self.words)
        self.guesses = tensor(self.guesses)

    def __getitem__(self, i):
        return self.words[i], self.guesses[i]

    def __len__(self):
        return self.words.__len__()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class HangmanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=28, embedding_dim=512, padding_idx=27)
        self.pos_enc = PositionalEncoding(d_model=512)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 4)

        self.head = nn.Sequential(nn.Linear(in_features=29*512, out_features=2048),
                                  nn.Linear(in_features=2048, out_features=512),
                                  nn.Linear(in_features=512, out_features=26))

    def forward(self, x):
        #x[0].to(DEVICE)
        #x[1].to(DEVICE)
        inputs, masks = x
        embeddings = self.embedder(inputs)
        positional_embeddings = self.pos_enc(embeddings)
        encodings = self.encoder(src=positional_embeddings, mask=masks)
        logits = self.head(encodings.flatten(1, 2))
        return logits

    def step(self, dl, optim, loss_fn):
        total_loss = 0
        for batch in tqdm(dl, total=len(dl)):
            inputs, guesses = batch
            inputs = inputs.to(DEVICE)
            guesses = guesses.to(DEVICE)
            s_mask = (inputs != 27)
            masks = torch.mul(s_mask.unsqueeze(2), s_mask.unsqueeze(1)).repeat(8, 1, 1)
            masks = masks.logical_not()
            logits = self((inputs, masks))
            loss = loss_fn(logits, guesses)
            loss.backward()
            optim.step()
            self.zero_grad()
            total_loss += loss.item()
        return total_loss / len(dl)
    
    def train_loop(self, dl):
        self.train()
        lr = 0.1
        optim = torch.optim.SGD(params=self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=27)

        min_loss = math.inf
        count = 0
        for e in tqdm(range(100)):
            loss = self.step(dl, optim, loss_fn)
            if loss < min_loss:
                min_loss = loss
                count = 0
                torch.save(self, "model.pkz")
            else:
                count += 1

            if count == 3: break

#ds = HangmanData()
#torch.save(ds, 'dataset.pkl')

ds = torch.load('dataset.pkl', map_location=DEVICE)
# Take 10% of dataset for memory reasons
perm = torch.randperm(len(ds))
idx = perm[:len(ds)//10]
ds.words = ds.words[idx]
ds.guesses = ds.guesses[idx]
dl = DataLoader(ds, batch_size=1024, shuffle=True)

hm = HangmanModel()
hm = hm.to(DEVICE)
hm.train_loop(dl)