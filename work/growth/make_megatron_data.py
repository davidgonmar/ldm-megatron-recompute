"""OLMo-core numpy token shards (flat uint16, documents ending in EOS) -> one Megatron indexed
dataset (.bin / .idx), one sequence per document.
    python3 make_megatron_data.py OUT_PREFIX SHARD.npy [SHARD.npy ...]
"""
import struct
import sys

import numpy as np

EOS = 50279
HEADER = b"MMIDIDX\x00\x00"
UINT16_CODE = 8  # megatron.core.datasets.indexed_dataset.DType.uint16

prefix, shards = sys.argv[1], sys.argv[2:]
lengths = []
with open(prefix + ".bin", "wb") as out:
    for path in shards:
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        ends = np.flatnonzero(tokens == EOS) + 1
        if ends.size == 0 or ends[-1] != tokens.size:
            ends = np.append(ends, tokens.size)
        lengths.append(np.diff(ends, prepend=0))
        out.write(tokens.tobytes())
        print(path, tokens.size, "tokens", ends.size, "documents", flush=True)
lengths = np.concatenate(lengths).astype(np.int32)
pointers = np.concatenate(([0], np.cumsum(lengths.astype(np.int64))[:-1])) * 2
with open(prefix + ".idx", "wb") as idx:
    idx.write(HEADER)
    idx.write(struct.pack("<Q", 1))
    idx.write(struct.pack("<B", UINT16_CODE))
    idx.write(struct.pack("<Q", lengths.size))
    idx.write(struct.pack("<Q", lengths.size + 1))
    idx.write(lengths.tobytes(order="C"))
    idx.write(pointers.astype(np.int64).tobytes(order="C"))
    idx.write(np.arange(lengths.size + 1, dtype=np.int64).tobytes(order="C"))
print("total", int(lengths.sum()), "tokens", lengths.size, "documents")
