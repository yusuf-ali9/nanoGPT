
import torch
import torch.nn as nn
import torch.nn.functional as F

#Hyperparameters
batch_size = 16
block_size = 64
eval_iters = 5
n_embd = 128
num_heads = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# learning_rate = 3e-4
learning_rate = 1e-3
n_layer = 6
droupout = 0.2
max_iters = 1000
eval_interval = 100


#Data loading
# !shakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()

#Create a set of unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#Create mappings from characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }


def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])
#Encode the entire text into a list of integers
data = torch.tensor(encode(text), dtype=torch.long)
#Split the data into training and validation sets
n = int(0.9*len(data))


train_data = data[:n]
val_data = data[n:]
#Data loading functions
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # train_data = train_data.to(device)
    # val_data = val_data.to(device)
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
#Model definition

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out






class BigramLanguageModel(nn.Module):   
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedd = nn.Embedding(block_size, n_embd)
    

        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #layer norm right before final layer
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape #B = 32, T = 8, n_embd = 32, vocab_size = 65
        tok_emb = self.token_embedding_table(idx) # (B,T, C)
        pos_emb = self.pos_embedd(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
       
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)# (B,T,vocab_size)   

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] 
            logits, loss = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.droupout = nn.Dropout(droupout)
    
    def forward(self, x):
        
        B, T, C = x.shape #C = n_embd, T = block_size
        k = self.key(x) #(n_embd, head_size) @ (B, T, n_embd) -> (B, T, head_size)
        q = self.query(x) #(n_embd, head_size) @ (B, T, n_embd) -> (B, T, head_size)
        w = q @ k.transpose(-2, -1) * C **-0.5 #(B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.droupout(w)

        v = self.value(x) #(n_embd, head_size) @ (B, T, n_embd) -> (B, T, head_size)
        output = w @ v #(B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return output
    
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #B, T, C
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(droupout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        headsize = n_embd // n_heads
        self.sa = MultiHead(n_heads, headsize)
        self.ffwd = FeedForward(n_embd)
        self.lnl1 = nn.LayerNorm(n_embd)
        self.lnl2 = nn.LayerNorm(n_embd)
        

    def forward(self, x):
        x = x + self.sa(self.lnl1(x))
        x = x + self.ffwd(self.lnl2(x))
        return x
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4), #Stretch by k = 4 as per paper to create larger non linearity work space before compressing
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(droupout)
        )
    def forward(self, x):
        x = self.net(x)
        return x
        
    

#Instantiate the model and move it to the appropriate device
model = BigramLanguageModel()
m = model.to(device)

#Define the loss function and optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#Training loop

for iter in range(max_iters):
# for iter in range(100):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
#Generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))