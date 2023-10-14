import numpy as np


def positional_encoding(max_seq_len: int, dim: int, n: int = 10_000):
    PE = np.zeros((max_seq_len, dim))
    for pos in range(max_seq_len):
        for i in range(dim // 2):
            den = n ** (2 * i / dim)
            PE[pos, 2 * i] = np.sin(pos / den)
            PE[pos, 2 * i + 1] = np.cos(pos / den)
    return PE
