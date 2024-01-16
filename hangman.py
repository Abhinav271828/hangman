from torch import nn, tensor
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from tqdm import tqdm
from random import sample, uniform

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
            self.guesses.append(ord(guess))

        print("Tensorifying")
        self.words = tensor(self.words)
        self.guesses = tensor(self.guesses)

    def __getitem__(self, i):
        return self.words[i], self.guesses[i]

    def __len__(self):
        return self.words.__len__()

#ds = HangmanData()
#torch.save(ds, 'dataset.pkl')

class HangmanModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def step(self):
        pass