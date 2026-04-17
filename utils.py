import numpy as np

NL = 8
def read(path, sessions=None):
    data = np.memmap(path, dtype=np.int32, mode="r").reshape(-1, NL * 6 + 1)
    if sessions is None:
        bounds = [(0, len(data))]
    else:
        b = np.fromfile(sessions, dtype=np.int64)
        bounds = list(zip(b[:-1], b[1:]))
    result = []
    for s, e in bounds:
        d = data[s:e]
        result.append((
            d[:, 0 * NL:1 * NL],
            d[:, 1 * NL:2 * NL],
            d[:, 2 * NL:3 * NL],
            d[:, 3 * NL:4 * NL],
            d[:, 4 * NL:5 * NL],
            d[:, 5 * NL:6 * NL],
            d[:, 6 * NL],
            np.arange(s, e, dtype=np.int64),
        ))
    return result

def direct(session):
    askRate, bidRate, askSize, bidSize, askNC, bidNC, y, tick = session
    for t in range(len(askRate)):
        yield (
            askRate[t].tolist(),
            bidRate[t].tolist(),
            askSize[t].tolist(),
            bidSize[t].tolist(),
            askNC[t].tolist(),
            bidNC[t].tolist(),
            y[t].item(),
            tick[t].item(),
        )


def head(session):
    askRate, bidRate, askSize, bidSize, askNC, bidNC, y, tick = session
    return (
            askRate[0].tolist(),
            bidRate[0].tolist(),
            askSize[0].tolist(),
            bidSize[0].tolist(),
            askNC[0].tolist(),
            bidNC[0].tolist(),
            y[0].item(),
            tick[0].item(),
        )

def tail(session):
    askRate, bidRate, askSize, bidSize, askNC, bidNC, y, tick = session
    return (
            askRate[-1].tolist(),
            bidRate[-1].tolist(),
            askSize[-1].tolist(),
            bidSize[-1].tolist(),
            askNC[-1].tolist(),
            bidNC[-1].tolist(),
            y[-1].item(),
            tick[-1].item(),
        )
