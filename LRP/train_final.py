from evaluator import train_and_eval

def train_best(code, epochs, seed, dataset):
    acc = train_and_eval(code, epochs=epochs, seed=seed, dataset=dataset)
    print(f"Final accuracy: {acc:.4f}")
    return acc
