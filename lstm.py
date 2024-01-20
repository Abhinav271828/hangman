# %%
from torch import nn, tensor
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from tqdm import tqdm
from random import sample, uniform
import math
from icecream import ic
from copy import deepcopy

# %%
DEVICE = torch.device('cpu')

# %%
class LSTMData(Dataset):
    def __init__(self):
        f = open('words_250000_train.txt', 'r')
        
        self.words = []
        self.uniques = []
        for line in tqdm(f, total=227300):
            word = line[:-1]
            self.words.append([ord(c)-ord('a') for c in word] + [27]*(29-len(word)))
            self.uniques.append([ord(c)-ord('a') for c in set(word)] + [27]*(16-len(set(word))))

        print("Tensorifying")
        self.words = tensor(self.words)
        self.uniques = tensor(self.uniques)

    def __getitem__(self, i):
        return self.words[i], self.uniques[i]

    def __len__(self):
        return self.words.__len__()

# %%
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=28, embedding_dim=128, padding_idx=27)

        self.encoder = nn.LSTM(input_size=128, hidden_size=128, num_layers=3, batch_first=True, bidirectional=True)

        self.head = nn.Linear(in_features=2*128, out_features=27)

    def forward(self, x):
        # This is masked
        inputs = x # [b, 29]

        embeddings = self.embedder(inputs)
        encodings, _ = self.encoder(embeddings) # [b, 29, 256]
        logits = self.head(encodings) # [b, 29, 27]
        return logits

    def epoch(self, dl, optim, loss_fn, train):
        total_loss = 0
        for batch in tqdm(dl, total=len(dl)):
            if train: self.zero_grad()
            inputs, uniques = batch
            # With probability 0.3, mask 15%
            # "   "   "    "   0.3, mask 50%
            # "   "   "    "   0.4, mask 80%
            r = uniform(0, 1)
            if (r < 0.3): p = 0.20
            elif (r < 0.6): p = 0.50
            else: p = 0.80

            pad_mask = inputs != 27
            mlm_mask = torch.rand((inputs.shape[0], 29)) < p
            mask = mlm_mask & pad_mask

            labels = inputs.masked_fill(mask.logical_not(), 27)

            inputs = inputs.masked_fill(mask, 26)
            logits = self(inputs) # [b, 29, 26]

            loss = loss_fn(logits.transpose(1, 2), labels)

            if train:
                loss.backward()
                optim.step()

            total_loss += loss.item()
        return total_loss / len(dl)
    
    def train_loop(self, train_dl, val_dl, optim=None):
        lr = 1e-3
        optim = optim or torch.optim.Adam(params=self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=27)

        min_loss = math.inf
        for e in tqdm(range(10000)):
            self.train()
            self.epoch(train_dl, optim, loss_fn, True)
            self.eval()
            val_loss = self.epoch(val_dl, optim, loss_fn, False)
            print(val_loss)
            if val_loss < min_loss:
                min_loss = val_loss
            state = {'epoch': e, 'state_dict': self.state_dict(),
             'optimizer': optim.state_dict(), 'loss': val_loss}
            torch.save(state, f"lstm-{e}-{val_loss:.4f}.ckpt")

# %%
train_ds = torch.load('lstmdataset.pkl', map_location=DEVICE)
val_ds = deepcopy(train_ds)

perm = torch.randperm(len(train_ds))
val_idx = perm[:len(train_ds)//20]
val_ds.words = train_ds.words[val_idx]
val_ds.uniques = train_ds.uniques[val_idx]

train_idx = perm[len(train_ds)//20:]
train_ds.words = train_ds.words[train_idx]
train_ds.uniques = train_ds.uniques[train_idx]

train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=1024, shuffle=True)

# %%
#hm = LSTMModel()
#hm = hm.to(DEVICE)
#hm.train_loop(train_dl, val_dl)

hm = LSTMModel()
hm.load_state_dict(torch.load('lowlr-235-1.522131.ckpt', map_location="cpu")['state_dict'])

optim = torch.optim.Adam(hm.parameters())
optim.load_state_dict(torch.load('lowlr-235-1.522131.ckpt', map_location="cpu")['optimizer'])

hm.train_loop(train_dl, val_dl, optim)