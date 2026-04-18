"""Preprocess events stream: split tm → {tm_queue, tm_cascade} based on pre-event N[0].

Reads train.events for one session via stdin or file slicing, writes binary
(t, pooled_type) stream to stdout for hawkes.c.

Pooled 6-D taxonomy:
   0 tp (ask+bid pooled)
   1 tm_queue  (tm fired when pre-event N[0] > 1 — no cascade)
   2 tm_cascade(tm fired when pre-event N[0] == 1 — cascade + refill)
   3 dp (ask+bid pooled)
   4 dm (ask+bid pooled)
   5 hp (ask+bid pooled; observation artifact)

Usage:
   python3 preproc_events.py <session_id> > events.bin
"""
import sys, numpy as np, struct

RECSZ = 216      # 54 int32 per record in train.events

ses = int(sys.argv[1])
ev = np.memmap('data/train.events', dtype=np.int32, mode='r').reshape(-1, 54)
offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)
lo, hi = int(offs[ses]) // RECSZ, int(offs[ses+1]) // RECSZ
block = ev[lo:hi]

types  = block[:, 0].astype(np.int32)
orig   = block[:, 1].astype(np.int32)
# Pre-event top-of-book counts. Book cols: aR(5..12), bR(13..20), aS(21..28),
# bS(29..36), aN(37..44), bN(45..52). So aN[0] = col 37, bN[0] = col 45.
aN0 = block[:, 37].astype(np.int32)
bN0 = block[:, 45].astype(np.int32)

# Compose output stream: (t, pooled_type)
out = []
n_q = n_c = 0
for i in range(len(block)):
    t = types[i]
    if t == 8: continue           # skip IDLE records
    if t == 0 or t == 1:
        out.append((int(orig[i]), 0))       # tp
    elif t == 2:                   # tm_a
        if aN0[i] > 1: out.append((int(orig[i]), 1)); n_q += 1
        else:          out.append((int(orig[i]), 2)); n_c += 1
    elif t == 3:                   # tm_b
        if bN0[i] > 1: out.append((int(orig[i]), 1)); n_q += 1
        else:          out.append((int(orig[i]), 2)); n_c += 1
    elif t == 4 or t == 5:
        out.append((int(orig[i]), 3))       # dp
    elif t == 6 or t == 7:
        out.append((int(orig[i]), 4))       # dm
    elif t == 9 or t == 10 or t == 11:      # hp (older encodings?)
        out.append((int(orig[i]), 5))

sys.stderr.write(f"ses{ses}: {len(out)} events  tm_q={n_q}  tm_c={n_c}  "
                 f"cascade frac = {n_c/max(n_q+n_c,1):.4f}\n")

buf = np.array(out, dtype=np.int32)
sys.stdout.buffer.write(buf.tobytes())
