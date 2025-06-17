import random
import torch
import numpy as np
from search_space import NASearchSpace
from predictor_trainer import PredictorTrainer
from evolutionary_operators import mask_crossover, polynomial_mutation
from evaluator import train_and_eval
from utils import set_seed

class EvolutionarySearch:
    def __init__(self, config):
        set_seed(config['seed'])
        self.config = config
        self.space = NASearchSpace()
        self.predictor = PredictorTrainer(config, self.space).train()
        self.pop = self.space.random_population(config['pop_size'])

    def evaluate(self, pop, use_pred=True):
        if use_pred:
            X = np.array(pop).reshape(1, len(pop), -1)
            with torch.no_grad():
                scores = self.predictor(torch.tensor(X, dtype=torch.float32)).squeeze(0).tolist()
            return scores
        else:
            return [train_and_eval(ind, epochs=self.config['eval_epochs'], seed=self.config['seed'], dataset=self.config['dataset']) for ind in pop]

    def environment_selection(self, parents, children):
        combined = parents + children
        scores = self.evaluate(combined, use_pred=True)
        idx = sorted(range(len(combined)), key=lambda i: scores[i], reverse=True)
        return [combined[i] for i in idx[:self.config['pop_size']]]

    def run(self):
        pop = self.pop
        # warm-up
        fitness = self.evaluate(pop, use_pred=False)
        for gen in range(self.config['generations']):
            parents = self.environment_selection(pop, [])
            children = []
            for _ in range(len(parents)//2):
                p1, p2 = random.sample(parents, 2)
                child = mask_crossover(p1, p2, self.config['crossover_prob'])
                child = polynomial_mutation(child, self.config['mutation_prob'], self.config['eta'])
                children.append(child)
            pop = self.environment_selection(parents, children)
        return pop[0]
