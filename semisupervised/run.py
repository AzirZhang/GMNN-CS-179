import yaml
from train import Train
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer')
args = parser.parse_args()

with open('./configs.yaml', 'r') as file:
    opt = yaml.safe_load(file)

opt['data'] = opt['data'] + f'{args.dataset}'
opt['decay'] = float(opt['decay'])

print(opt['data'])

for k in range(100):
    seed = random.randint(0, 2048)
    opt['save'] = f'./data/{args.dataset}/params/direc{k}'
    opt['seed'] = seed
    t = Train(opt)
    t.train()

