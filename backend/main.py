import torch
import torch.nn as nn
import torch.nn.functional as F
# import tensorflow as tf

# encoder receives an input (series of text tokens) and builds a representation of its features
# trained to acquire an understanding
# batch dimension
torch.manual_seed(1337) 
batch_size = 4 # number of indepedent sequences we process in parallel - every forward/backward pass in transformer
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_batch, y_batch = get_batch('train')
print("inputs: ")
print(x_batch.shape)
print(x_batch)
print("targets: ")
print(y_batch.shape)
print(y_batch)

# here we essentially need to load the data - sequences of chords or notes
# chord library should 
chord_library = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
chords = sorted(list(set(chord_library)))
chord_library_size = len(chords)
# later, load into tensors

# create a mapping from chords to integers
stoi = { ch:i for i, ch in enumerate(chords) }
itos = { i:ch for i, ch in enumerate(chords) }
encode = lambda s: [stoi[c] for c in s] # this is our encoder. it takes a chord string 
# from our library and outputs a list of integers using stoi[c] for each c in "s", s being the input chord string
decode = lambda l: ''.join([itos[i] for i in l]) # decoder take list of integers -> output chord string
# train and test splits

data = torch.tensor(encode(chord_library), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n] # first 90% of data/from the start to "n", n being 90%
val_data = data[n:] # from n (90% to the end - merely a convention)

# FUNCTION DEFINITIONS

def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generate random positions from which to take chunks - random offsets from 0 to len(data) - block_size
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # x and y are now both (batch_size, block_size) tensors
    # for now, 4x8 tensor
    x, y = x.to(device), y.to(device)
    return x, y

# xb, yb = get_batch('train')

# simplest possible neural network - a Bigram language model
# in essence, its a statistial language model in NLP that predicts the likelihood of a word (chord!?) in a sequence based on the preceding word (chord!?!?)
# class BigramLanguageModel(nn.Module):
#     def __init__(self, chord_library_size):
#         # super() - gives access to parent class' methods
#         super().__init__()
#         # each token directly reads off the "logits" for the next token from the lookup table
#         # define a matrix/look up table such that each row is a dense vector that represents a token from the model - possibly a chord or note/frequency
#         # Embedding - wrapper aroudn tensor of chord_lib_size by chord_lib_size
#         self.token_embedding_table = nn.Embedding(chord_library_size, chord_library_size)

#     def forward(self, idx, targets):

#         # idx and targets are both (B, T) tensor of integers?
#         logits = self.token_embedding_table(idx) # (B, T, C) - batch by time by channel.
        
#         # now, we want to evaluate a "loss" function/quality of predictions

#         B, T, C = logits.shape
#         logits = logits.view(B*T, C) # so now we're stretching out the array such that its 2D to conform to what pytorch expects for loss_entropy
#         targets = targets.view(B*T) # flatten to 1D

#         loss = F.cross_entropy(logits, targets) # - how well are we predicting the next chord?
        
#         # cross_entropy in its "functional" form - pytorch expects the second input of multidimensional inputs to be the "channels"
#         # so we actually have to reshape our logits
#         return logits, loss # scores for next chord in sequence - predicting what comes next based on the INDIVIDUAL identity of a single token. each token sees themselves
    
# m = BigramLanguageModel(chord_library_size)
# logits, loss = m(xb, yb)

# print(logits.shape)
# print(f"Loss: {loss}")
# decoder uses those representations alongside other inputs to generate a sequence
# trained to generate outputs

# now need audio

# mp3 -> WAV -> Spectrogram  -> pitch detection -> note grouping -> tokenization -> transformer model -> prediction

# ffmpeg or py- audio decoding
# numpy - feature extraction
# deep learning - torch
# transformer weights - transformers
# visualization - matplotlib

# MP3 -> waveform
# pytorch, torchaudio, librosa?, 
# pyaudio captures live audio from a microphone

# pyaudio -> torchaudio -> feature extraction? -> transformer -> flask API w/ endpoint
# madmom apparently has pre-trained models for musical genres
