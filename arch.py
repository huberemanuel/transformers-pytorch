import torch
from torch import nn

from utils import positional_encoding


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int = 512, heads: int = 8):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        if self.head_dim * heads != embed_size:
            raise ValueError(f"Cannot split {embed_size} equally into {heads}")

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = self.values(values)  # values.shape = (batch, max_seq_len, embed_size)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # TODO: Check if this is equivalent to projecting each q,k,v to separate W matrices and
        # them concatenating them back?
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 1, 3)

        scores = torch.matmul(queries, keys)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))

        # TODO: Test with dim=2, since Q and V come from the source sentence.
        scores = torch.softmax(scores / (self.embed_size ** (1 / 2)), dim=-1)
        attention = torch.matmul(scores, values).reshape(N, query_len, self.embed_size)

        out = self.fc_out(attention)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size: int = 512,
        heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.multi_head_attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, value, key, query, mask=None):
        out = self.multi_head_attention(value, key, query, mask)
        x = self.norm1(x + self.drop1(out))
        out = self.fc2(self.fc1(x).relu())
        out = self.norm2(x + self.drop2(out))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        num_layers: int = 6,
        heads: int = 8,
        device: str = "cuda",
        max_seq_len: int = 100,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.token_embedding = Embedding(
            vocab_size=vocab_size,
            dim=embed_size,
            max_seq_len=max_seq_len,
            n=10_000,
            dropout=dropout,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size, heads=heads, ff_dim=ff_dim, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        x = x.to(self.device)
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x=x, value=x, key=x, query=x, mask=mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int = 512,
        heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.drop = nn.Dropout(dropout)
        self.transformer_block = TransformerBlock(
            embed_size=embed_size, heads=heads, ff_dim=ff_dim, dropout=dropout
        )
        self.multi_head_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        out = self.multi_head_attention(x, x, x, tgt_mask)
        x = self.norm(x + self.drop(out))
        out = self.transformer_block(
            x=x, value=enc_out, key=enc_out, query=x, mask=src_mask
        )
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        embed_size: int = 512,
        num_layers: int = 6,
        heads: int = 8,
        device: str = "cuda",
        max_seq_len: int = 100,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.token_embedding = Embedding(
            vocab_size,
            embed_size,
            device=device,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size=embed_size, heads=heads, dropout=dropout, ff_dim=ff_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = x.to(self.device)
        x = self.token_embedding(x)

        for layer in self.layers:
            x = layer(x=x, enc_out=enc_out, src_mask=src_mask, tgt_mask=tgt_mask)

        out = self.fc(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 1000,
        tgt_vocab_size: int = 1000,
        embed_size: int = 512,
        num_layers: int = 6,
        heads: int = 8,
        device: str = "cuda",
        max_seq_len: int = 100,
        dropout: float = 0.1,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        ff_dim: int = 2048,
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            max_seq_len=max_seq_len,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            max_seq_len=max_seq_len,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n: int = 10_000,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.embed = nn.Embedding(vocab_size, dim)
        self.pe = torch.tensor(positional_encoding(self.max_seq_len, self.dim)).to(
            device
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x)
        x += self.pe[x.shape[0]]
        return self.drop(x)
