import yaml
from train import Train
import random
import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer')
parser.add_argument('--num_exp', type=int, default=10)
args = parser.parse_args()

with open(f'configs_{args.dataset}.yaml', 'r') as file:
    opt = yaml.safe_load(file)

opt['data'] = opt['data'] + f'{args.dataset}'
opt['decay'] = float(opt['decay'])

print(opt['data'])

scores = torch.zeros(5)
num_exp = args.num_exp
tt = time.time()
for k in range(num_exp):
    seed = random.randint(0, 2048)
    opt['save'] = f'./data/{args.dataset}/params/direc{k}'
    opt['seed'] = seed
    st = time.time()
    t = Train(opt)
    pr, qr = t.train()
    best_scores = torch.max(qr[0], dim=0)[0]
    scores += best_scores
    # if (k+1) % 5 == 0:
    #     m = f'\n  '.join([c.capitalize() + ": " + str(float(best_scores[i])) for i, c in enumerate(opt['metric'])])
    #     print(f"Experiment {k+1}\n  Time: {time.time()-st}\n  {m}")

total_scores = scores/num_exp
m = f'\n  '.join([c.capitalize() + ": " + str(float(total_scores[i])) for i, c in enumerate(opt['metric'])])
print(f"Experiment Summary\n  Time: {time.time()-tt}\n  Number of Experiments: {num_exp}\n  {m}")
