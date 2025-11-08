import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

# encoder receives an input (series of text tokens) and builds a representation of its features
# trained to acquire an understanding
# batch dimension
# hyperparameters

# CHANGE HYPERPARAMETERS TO SMALLER VALUES IF RUNNING LOCALLY
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)
# tensor of encoded integers that represent that text; dtype (datatype) is long
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:] # from n to end, n being the first 90% of the data

def get_batch(split): # get a random batch of data
    if split == 'train':
        data = train_data
    else:
        data = val_data
    print("get_batch running")
    # ix is a tensor of random integers
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # x and y will then be "stacks" - tensors of shape (batch_size, block_size), both being hyperparameters
    # this is essentially generating random chunks of data from the text
    # batch_size - how many sequences processed in parallel
    # block_size - how many characters/notes/chords in each sequence
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print("running iteration:", iter)
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    print("getting batch")
    xb, yb = get_batch('train')
    print("got batch")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print("completed step")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

"""
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
"""