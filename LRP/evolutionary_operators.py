import random

def mask_crossover(p1, p2, prob):
    return [a if random.random()<prob else b for a,b in zip(p1,p2)]

def polynomial_mutation(code, prob, eta):
    new = code.copy()
    L = len(code)
    for i in range(L):
        if random.random() < prob:
            delta = (random.random()*2-1)*(1/eta)
            new[i] = max(0, int(new[i] + delta*(max(code)+1)))
    return new
