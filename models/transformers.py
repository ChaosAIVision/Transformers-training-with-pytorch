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
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.sa_layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)  # Self-Attention Layer Normalization
        self.ffn = FFN(embed_dim, ff_dim, dropout)
        self.output_layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)  # Output Layer Normalization

    def forward(self, x, mask):
        # Self-Attention and residual connection
        attn_output = self.attention(x, x, x, mask)  # (batch_size, seq_len, d_model)
        x = self.sa_layer_norm(x + attn_output)  # Residual connection and layer normalization

        # Feed-Forward Network and residual connection
        ffn_output = self.ffn(x)  # (batch_size, seq_len, d_model)
        x = self.output_layer_norm(x + ffn_output)  # Residual connection and layer normalization

        return x


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, dropout, device):
        super().__init__()
        self.device = device
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embeddings = nn.Embedding(num_embeddings=max_length, embedding_dim=embed_dim)
        self.Layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
        word_emb = self.word_embeddings(x)
        pos_emb = self.position_embeddings(positions)
        x = word_emb + pos_emb
        return self.Layer_norm(self.dropout(x))

class TransformerTextCls(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim, num_heads, ff_dim, dropout, device):
        super().__init__()
        self.embd_layer = TokenAndPositionEmbedding(vocab_size, embed_dim, max_length, dropout, device)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(6)  # Number of transformer layers
        ])
        self.fc1 = nn.Linear(embed_dim, 768)
        self.fc2 = nn.Linear(768, 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.embd_layer(x)
        for layer in self.transformer_layers:
            x = layer(x, mask)
        x = self.dropout(x.mean(dim=1))  # Pooling
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    vocab_size = 30522  # Kích thước từ vựng, ví dụ như kích thước từ vựng của DistilBERT
    max_length = 128    # Độ dài tối đa của chuỗi đầu vào
    embed_dim = 768     # Kích thước của các embedding vectors, ví dụ như kích thước embedding của DistilBERT
    num_heads = 12      # Số lượng đầu của multi-head attention
    ff_dim = 3072       # Kích thước của feed-forward network, tương tự như DistilBERT
    dropout = 0.1       # Tỷ lệ dropout

    # Chọn thiết bị (CPU hoặc GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Khởi tạo mô hình
    model = TransformerTextCls(
        vocab_size=vocab_size,
        max_length=max_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        device=device
    ).to(device)
    # Các tham số của mô hình
    batch_size = 2  # Số lượng mẫu trong mỗi batch
    seq_len = max_length  # Độ dài của chuỗi đầu vào

    # Tạo dữ liệu giả lập
    # Dữ liệu đầu vào giả lập với giá trị ngẫu nhiên trong phạm vi vocab_size
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Tạo mask giả lập (nếu cần). Trong ví dụ này, mask được đặt là None.
    dummy_mask = None

    # Tiến hành dự đoán
    model.eval()  # Đặt mô hình ở chế độ đánh giá (evaluation mode)
    with torch.no_grad():
        output = model(dummy_input, dummy_mask)

    # In kích thước của đầu ra để kiểm tra
    print("Output size:", output.size())

