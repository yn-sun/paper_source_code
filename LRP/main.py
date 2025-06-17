import argparse
from config import config
from evolutionary_search import EvolutionarySearch
from train_final import train_best

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=config['dataset'])
    parser.add_argument('--predictor', type=str, default=config['predictor'])
    args = parser.parse_args()
    config['dataset'] = args.dataset
    config['predictor'] = args.predictor

    es = EvolutionarySearch(config)
    best = es.run()
    print("Best code:", best)
    train_best(best, epochs=config['final_epochs'], seed=config['seed'], dataset=config['dataset'])
