import torch

from arch import (
    Decoder,
    DecoderBlock,
    Embedding,
    Encoder,
    SelfAttention,
    Transformer,
    TransformerBlock,
)


def test_decoder():
    max_seq_len = 5
    vocab_size = 1000
    enc = Decoder(
        vocab_size=vocab_size,
        embed_size=512,
        num_layers=6,
        heads=8,
        device="cpu",
        max_seq_len=max_seq_len,
    )

    batch = 2
    input = torch.arange(batch * max_seq_len).view(batch, max_seq_len)
    x = torch.zeros((batch, max_seq_len, 512))
    out = enc(input, x)
    assert list(out.shape) == [batch, max_seq_len, vocab_size]


def test_decoder_block():
    batch = 2
    max_seq_len = 5
    embed_size = 512
    heads = 8
    block = DecoderBlock(embed_size=embed_size, heads=heads)
    x = torch.zeros((batch, max_seq_len, embed_size))
    enc_out = torch.zeros((batch, max_seq_len, embed_size))
    out = block(x, enc_out)
    assert list(out.shape) == [batch, max_seq_len, embed_size]


def test_encoder():
    max_seq_len = 5
    batch = 2
    enc = Encoder(
        vocab_size=1000,
        embed_size=512,
        num_layers=6,
        heads=8,
        device="cpu",
        max_seq_len=max_seq_len,
    )

    input = torch.arange(batch * max_seq_len).view(batch, max_seq_len)
    out = enc(input)
    assert list(out.shape) == [batch, max_seq_len, 512]


def test_self_attention():
    embed_size = 512
    heads = 8
    att = SelfAttention(embed_size, heads)
    print(att)

    N = 16
    max_seq_len = 100
    key = torch.zeros((N, max_seq_len, embed_size))
    values = torch.zeros((N, max_seq_len, embed_size))
    query = torch.zeros((N, max_seq_len, embed_size))

    out = att(values, key, query)
    assert list(out.shape) == [N, max_seq_len, embed_size]


def test_transformer_block():
    block = TransformerBlock(embed_size=512, heads=8)
    t = torch.zeros((16, 100, 512))
    out = block(t, t, t, t)
    assert list(out.shape) == [16, 100, 512], out.shape


def test_transformer():
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2, 0]])
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0, 0, 0], [1, 5, 6, 2, 4, 7, 6, 2, 0, 0]])

    src_pad_idx = 0
    tgt_pad_idx = 0
    src_vocab_size = 50_000
    tgt_vocab_size = 50_000
    embed_size = 512
    num_layers = 6
    heads = 8
    max_seq_len = 10
    device = "cpu"
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        device=device,
        max_seq_len=max_seq_len,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
    ).to(device)
    out = model(x, trg)
    assert list(out.shape) == [2, max_seq_len, tgt_vocab_size]


def test_embedding():
    batch = 2
    vocab_size = 30
    embed_dim = 4
    max_seq_len = 10
    device = "cpu"
    emb = Embedding(vocab_size, embed_dim, max_seq_len=max_seq_len, device=device)
    x = torch.arange(20).view(batch, max_seq_len)
    y = emb(x)
    assert list(y.shape) == [batch, max_seq_len, embed_dim]
