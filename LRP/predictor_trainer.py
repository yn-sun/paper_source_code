import torch
from torch.utils.data import DataLoader, TensorDataset
from rec_sampler import rec_construct
from predictor_model import ListNet, MLP, RFWrapper
from evaluator import train_and_eval
from utils import set_seed
import numpy as np
import random
import torch.nn.functional as F

class PredictorTrainer:
    def __init__(self, config, space):
        self.config = config
        self.space = space
        set_seed(config['seed'])
        model_type = config['predictor']
        D = len(space.random_population(1)[0])
        if model_type == 'listnet':
            self.model = ListNet(D, config['hidden_dim'])
            self.use_nn = True
        elif model_type == 'mlp':
            self.model = MLP(D, config['mlp_hidden'])
            self.use_nn = True
        else:
            self.model = RFWrapper(config['rf_trees'])
            self.use_nn = False

    def prepare_pairs(self):
        pairs = []
        for _ in range(self.config['num_pairs']):
            code = self.space.encode(self.space.sample_arch())
            acc = train_and_eval(code, epochs=self.config['eval_epochs'], seed=random.randint(0,10000), dataset=self.config['dataset'])
            pairs.append((code, acc))
        return pairs

    def train(self):
        pairs = self.prepare_pairs()
        X, Y = rec_construct(pairs, self.config['K'], self.config['M'])
        if self.use_nn:
            X_t = torch.tensor(X)
            Y_t = torch.tensor(Y)
            dataset = TensorDataset(X_t, Y_t)
            loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
            optim = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
            for epoch in range(self.config['epochs']):
                for xb, yb in loader:
                    pred = self.model(xb)
                    loss = ((yb - pred)**2).sum(dim=1).mean()
                    optim.zero_grad(); loss.backward(); optim.step()
        else:
            self.model.fit(X, Y)
        return self.model
