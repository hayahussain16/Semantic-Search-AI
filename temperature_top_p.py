import numpy as np

tokens = [" Paris", " Lyon", " the", " a", " beautiful"]

logits = np.array([5.0, 2.5, 2.0, 1.5, 1.0])


def softmax_t(logits, T):
    z = logits / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def top_p_filter(probs, p=0.9):
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    keep = order[: np.searchsorted(cum, p) + 1]

    out = np.zeros_like(probs)
    out[keep] = probs[keep]

    return out / out.sum()


for T in [0.2, 1.0, 2.0]:
    probs = softmax_t(logits, T)
    print(f"T={T}:", dict(zip(tokens, np.round(probs, 3))))


print(
    "top-p 0.9 at T=1:",
    dict(
        zip(
            tokens,
            np.round(
                top_p_filter(softmax_t(logits, 1.0)),
                3
            )
        )
    )
)