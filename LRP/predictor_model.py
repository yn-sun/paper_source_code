import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor

class ListNet(nn.Module):
    def __init__(self, D, hidden):
        super().__init__()
        self.fc1 = nn.Linear(D, hidden)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        b, K, D = x.shape
        x = x.view(b*K, D)
        h = F.relu(self.fc1(x))
        s = self.fc2(h).view(b, K)
        return F.softmax(s, dim=1)

class MLP(nn.Module):
    def __init__(self, D, hidden_dims):
        super().__init__()
        layers = []
        prev = D
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        b, K, D = x.shape
        x = x.view(b*K, D)
        s = self.net(x).view(b, K)
        return s

class RFWrapper:
    def __init__(self, trees=100):
        self.model = RandomForestRegressor(n_estimators=trees)
    def fit(self, X, Y):
        M, K, D = X.shape
        self.model.fit(X.reshape(M*K, D), Y.reshape(M*K))
    def predict(self, X):
        M, K, D = X.shape
        preds = self.model.predict(X.reshape(M*K, D))
        return preds.reshape(M, K)
