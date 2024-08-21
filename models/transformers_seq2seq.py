import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        assert d_model % h == 0, "d_model is not divisible by h"
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.q_lin = nn.Linear(d_model, d_model, bias=True)  # Wq
        self.k_lin = nn.Linear(d_model, d_model, bias=True)  # Wk
        self.v_lin = nn.Linear(d_model, d_model, bias=True)  # Wv
        self.out_lin = nn.Linear(d_model, d_model, bias=True)  # Wo

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.q_lin(q)
        key = self.k_lin(k)
        value = self.v_lin(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        return self.out_lin(x)

class FFN(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super(FFN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(embed_dim, ff_dim)
        self.lin2 = nn.Linear(ff_dim, embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.sa_layer_norm = nn.LayerNorm(d_model, eps=1e-12)  # Self-Attention Layer Normalization
        self.ffn = FFN(d_model, ff_dim, dropout)
        self.output_layer_norm = nn.LayerNorm(d_model, eps=1e-12)  # Output Layer Normalization

    def forward(self, x, mask):
        # Self-Attention and residual connection
        attn_output = self.attention(x, x, x, mask)  # (batch_size, seq_len, d_model)
        x = self.sa_layer_norm(x + attn_output)  # Residual connection and layer normalization

        # Feed-Forward Network and residual connection
        ffn_output = self.ffn(x)  # (batch_size, seq_len, d_model)
        x = self.output_layer_norm(x + ffn_output)  # Residual connection and layer normalization

        return x

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_length, dropout, device):
        super().__init__()
        self.device = device
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.position_embeddings = nn.Embedding(num_embeddings=max_length, embedding_dim=d_model)
        self.Layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
        word_emb = self.word_embeddings(x)
        pos_emb = self.position_embeddings(positions)
        x = word_emb + pos_emb
        return self.Layer_norm(self.dropout(x))


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_length, d_model, num_heads, ff_dim, dropout, device):
        super().__init__()
        self.embd_layer = TokenAndPositionEmbedding(vocab_size, d_model, max_length, dropout, device)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(6)  # Number of transformer layers
        ])
        
    def forward(self, x, mask=None):
        x = self.embd_layer(x)
        for layer in self.transformer_layers:
            x = layer(x, mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_length, d_model, num_heads, ff_dim, dropout, device):
        super().__init__()
        self.embd_layer = TokenAndPositionEmbedding(vocab_size, d_model, max_length, dropout, device)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(3)  # Number of transformer layers
        ])
        self.projection = ProjectionLayer(d_model, vocab_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embd_layer(x)
        for layer in self.transformer_layers:
            x = layer(x, tgt_mask)
            # Add cross attention layer
            x, _ = MultiHeadAttention.attention(x, enc_output, enc_output, src_mask, None)
        return self.projection(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, max_length, d_model, num_heads, ff_dim, dropout, device):
        super().__init__()
        self.d_model = d_model
        self.encoder = TransformerEncoder(vocab_size, max_length, d_model, num_heads, ff_dim, dropout, device)
        self.decoder = TransformerDecoder(vocab_size, max_length, d_model, num_heads, ff_dim, dropout, device)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output

if __name__ == "__main__":
    vocab_size = 30522  # Kích thước từ vựng
    max_length = 128    # Độ dài tối đa của chuỗi đầu vào
    d_model = 768       # Kích thước của các embedding vectors
    num_heads = 12      # Số lượng đầu của multi-head attention
    ff_dim = 3072       # Kích thước của feed-forward network
    dropout = 0.1       # Tỷ lệ dropout

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        device=device
    ).to(device)
    print(model)
    statedict = model.state_dict()
    for key in statedict:
        print(key)


    batch_size = 2  
    seq_len = max_length  

    dummy_input_src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    dummy_input_tgt = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    dummy_mask = None

    model.eval()  
    with torch.no_grad():
        output = model(dummy_input_src, dummy_input_tgt, dummy_mask, dummy_mask)

    print("Output size:", output.size())
