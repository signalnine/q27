"""Build a proposal-only Qwen3.8 vocabulary shortlist from token counts.

Input counts: little-endian int64 rows of --vocab elements; row zero contains
total token frequencies. Output: unique little-endian int32 original token IDs,
with all tokenizer special tokens retained. No model weights are modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('counts', type=Path)
    ap.add_argument('tokenizer_config', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--rows', type=int, default=131072)
    ap.add_argument('--vocab', type=int, default=248320)
    ap.add_argument('--tokenizer-vocab', type=int, default=248077)
    args = ap.parse_args()
    if not (128 <= args.rows <= args.tokenizer_vocab <= args.vocab) or args.rows % 128:
        ap.error('rows must be a multiple of 128 within the tokenizer vocabulary')
    size = args.counts.stat().st_size
    if not size or size % (args.vocab * 8):
        ap.error('counts must contain complete int64 vocabulary rows')
    counts = np.fromfile(args.counts, dtype='<i8', count=args.vocab)[:args.tokenizer_vocab]
    if np.any(counts < 0):
        ap.error('token counts must be nonnegative')
    cfg = json.loads(args.tokenizer_config.read_text())
    forced = np.array(sorted(int(token) for token, meta in cfg.get('added_tokens_decoder', {}).items()
                             if isinstance(meta, dict) and meta.get('special')), dtype=np.int64)
    if len(forced) > args.rows or np.any(forced < 0) or np.any(forced >= args.tokenizer_vocab):
        ap.error('special tokens do not fit the selected tokenizer domain')
    ids = np.arange(args.tokenizer_vocab, dtype=np.int64)
    order = np.lexsort((ids, -counts))
    special = np.zeros(args.tokenizer_vocab, dtype=bool)
    special[forced] = True
    selected = np.concatenate((order[~special[order]][:args.rows - len(forced)], forced))
    selected = selected[np.lexsort((selected, -counts[selected]))].astype('<i4')
    assert len(selected) == args.rows and len(np.unique(selected)) == args.rows
    data = selected.tobytes()
    with args.output.open('xb') as f:
        f.write(data)
    total = sum(map(int, counts))
    covered = sum(map(int, counts[selected]))
    print(json.dumps(dict(rows=len(selected), special_tokens=len(forced),
                          count_coverage=covered / total if total else None,
                          sha256=hashlib.sha256(data).hexdigest(), output=str(args.output))))


if __name__ == '__main__':
    main()
