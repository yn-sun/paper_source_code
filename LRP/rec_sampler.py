import random
import numpy as np

def rec_construct(pairs, K, M):
    X = np.zeros((M, K, len(pairs[0][0])), dtype=np.float32)
    Y = np.zeros((M, K), dtype=np.float32)
    for m in range(M):
        sampled = random.sample(pairs, K)
        codes, accs = zip(*sampled)
        X[m] = np.array(codes)
        scores = np.array(accs)
        exp = np.exp(scores - np.max(scores))
        Y[m] = exp / exp.sum()
    return X, Y
