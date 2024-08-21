import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads        
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.out_lin = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def attention(self, query, key, value, mask):
        # Scale the dot product by the dimensions of the key
        scaled_attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            scaled_attention_scores += (mask * -1e9)
        attention_weights = F.softmax(scaled_attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q = self.split_heads(self.q_lin(q), batch_size)
        k = self.split_heads(self.k_lin(k), batch_size)
        v = self.split_heads(self.v_lin(v), batch_size)

        attention_output, attention_weights = self.attention(q, k, v, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_lin(attention_output)
        return output, attention_weights




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
        self.sa_layer_norm = nn.LayerNorm(d_model, eps=1e-12)  # Chuẩn hóa lớp Self-Attention
        self.ffn = FFN(d_model, ff_dim, dropout)
        self.output_layer_norm = nn.LayerNorm(d_model, eps=1e-12)  # Chuẩn hóa lớp đầu ra

    def forward(self, x, mask):
        # Self-Attention và residual connection
        attn_output, _ = self.attention(x, x, x, mask)  # (batch_size, seq_len, d_model)
        x = self.sa_layer_norm(x + attn_output)  # Residual connection và chuẩn hóa lớp

        # Mạng Feed-Forward và residual connection
        ffn_output = self.ffn(x)  # (batch_size, seq_len, d_model)
        x = self.output_layer_norm(x + ffn_output)  # Residual connection và chuẩn hóa lớp

        return x


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
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
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embd_layer(x)
        for layer in self.transformer_layers:
            x = layer(x, tgt_mask)
            # Add cross attention layer
            cross_attn_output, _ = self.attention(x, enc_output, enc_output, src_mask)
            x = x + cross_attn_output  # Residual connection after cross attention
            x = self.layer_norm(x)  # Apply layer normalization after residual connection
        return x  # Trả về kích thước (batch_size, seq_len, d_model)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, max_length, d_model, num_heads, ff_dim, dropout, device):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, max_length, d_model, num_heads, ff_dim, dropout, device)
        self.decoder = TransformerDecoder(vocab_size, max_length, d_model, num_heads, ff_dim, dropout, device)
        self.projection = nn.Linear(d_model, vocab_size)  # Projection layer

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.projection(dec_output)  # Dùng lớp projection để chuyển đổi từ d_model -> vocab_size




if __name__ == "__main__":
    def create_padding_mask(seq):
    # Tạo mask cho các padding token (giả sử padding token là 0)
        return (seq == 0).unsqueeze(1).unsqueeze(2)  # Kích thước: (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(size):
        # Tạo mask chống xem trước
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask  # Kích thước: (seq_len, seq_len)

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

    dummy_input_src = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)  # Tránh 0 vì nó là padding token
    dummy_input_tgt = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)

    src_mask = create_padding_mask(dummy_input_src)
    tgt_mask = create_padding_mask(dummy_input_tgt) & create_look_ahead_mask(seq_len).to(device)

    model.eval()  # Đặt mô hình vào chế độ đánh giá
    with torch.no_grad():
        output = model(dummy_input_src, dummy_input_tgt, src_mask, tgt_mask)

    print("Output size:", output.size())


