import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loader
from search_space import NASearchSpace
from utils import set_seed

def train_and_eval(code, epochs=5, seed=0, dataset='cifar10', device=None):
    set_seed(seed)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loader(dataset)
    model = NASearchSpace().build_model(code).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    # evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total
