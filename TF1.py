import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as f
import pandas as pd
import torch.optim as optim
from tqdm.notebook import tqdm, trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device =='cuda':
    print("Train on GPU...")
else:
    print("Train on CPU...")
    
# Positional Encoding
def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim / dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
  
# Compute Attention(Q,K,V)
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

# 1-head Attention
class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

# Multi-head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )
    
# Feed Forward
def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

# Add & Norm
class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))
      
# Encoder Block
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)

# Encoder
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src
 
class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(tgt, memory, memory)
        return self.feed_forward(tgt)

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)
 
class Transformer(nn.Module):
    def __init__(
        self, 
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 27, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.2, 
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.decoder(tgt, self.encoder(src))
      
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from google.colab import drive
import string

drive.mount('/content/gdrive')

d_model = 32

# Data prep
big_data = pd.read_csv('/content/gdrive/MyDrive/824dat.csv')
data = big_data.iloc[:10000]
data['9core'] = data['9core'] + '------'
small_df = pd.DataFrame({'src': data.iloc[:, 0], 'tgt': data.iloc[:, -1]})
X_train, X_test, y_train, y_test = train_test_split(small_df['src'], small_df['tgt'], test_size=0.2, random_state=1)
train_df = pd.DataFrame({'src': X_train, 'tgt': y_train})
test_df = pd.DataFrame({'src': X_test, 'tgt': y_test})

# Alphabet
capital_letters = list(string.ascii_uppercase)
capital_letters.append('-')

def data_prep(d):
    src = d.src
    tgt = d.tgt

    # Convert strings to a list of list of ASCII values
    src_one_hot = []
    for row in src:
      src_one_hot_row = []
      for c in row:
        one_hot_encoding = torch.zeros(len(capital_letters))
        character_index = capital_letters.index(c)
        one_hot_encoding[character_index] = 1
        src_one_hot_row.append(one_hot_encoding)
      src_one_hot.append(src_one_hot_row)
    
    tgt_one_hot = []
    for row in tgt:
      tgt_one_hot_row = []
      for c in row:
        one_hot_encoding = torch.zeros(len(capital_letters))
        character_index = capital_letters.index(c)
        one_hot_encoding[character_index] = 1
        tgt_one_hot_row.append(one_hot_encoding)
      tgt_one_hot.append(tgt_one_hot_row)

    # Convert list of lists into tensor
    tensor_src = torch.stack([torch.stack(list_of_tensors) for list_of_tensors in src_one_hot])
    tensor_tgt = torch.stack([torch.stack(list_of_tensors) for list_of_tensors in tgt_one_hot])
        
    return tensor_src , tensor_tgt

s_train, t_train = data_prep(train_df)
s_test, t_test = data_prep(test_df)

# Create data loaders
trainset = TensorDataset(s_train,t_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = 100, shuffle = True)
testset = TensorDataset(s_test,t_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size = 100, shuffle = False)
model = Transformer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

acc_list = []

for epoch in trange(2):
    for src, tgt in tqdm(train_loader):

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        out = model(src, tgt) #torch.Size([100, 15, 27])
        loss = criterion(out, tgt)

        # Backward pass
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    for epoch in trange(2):
        correct = 0
        for src, tgt in tqdm(test_loader):
            out = model(src, tgt)
            loss = criterion(out, tgt)
            pred = torch.argmax(out, dim=-1)
            tgt_ch = torch.argmax(tgt, dim=-1)
            print(pred) #(100,15)
            print(tgt_ch) #(100,15)
            #print((pred == tgt_ch).float().sum())
        #avg_acc = correct/2000
        #acc_list.append(avg_acc)
