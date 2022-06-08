import yaml
from train import Train
import random
import torch
import time
import argparse
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer')
parser.add_argument('--num_exp', type=int, default=10)
parser.add_argument('--cpu', type=bool, default=False)
args = parser.parse_args()

with open(f'configs_{args.dataset}.yaml', 'r') as file:
    opt = yaml.safe_load(file)

opt['cpu'] = args.cpu
opt['data'] = opt['data'] + f'{args.dataset}'
opt['decay'] = float(opt['decay'])
opt['hidden_dim'] = 16
# opt['pre_epoch'] += 100
opt['epoch'] = 20

print(opt['data'])

num_exp = args.num_exp
scores = torch.zeros(opt['epoch'], 5)
score = torch.zeros(5)
tt = time.time()
for k in range(num_exp):
    seed = random.randint(0, 2048)
    opt['save'] = f'./data/{args.dataset}/params/direc{k}'
    opt['seed'] = seed
    st = time.time()
    t = Train(opt)
    pr, qr = t.train()
    # print(qr.shape)
    best_score = torch.max(qr[:, 0, :], dim=0)[0]
    score += best_score
    scores += qr[:, 0, :]
    # if (k+1) % 5 == 0:
    #     m = f'\n  '.join([c.capitalize() + ": " + str(float(best_scores[i])) for i, c in enumerate(opt['metric'])])
    #     print(f"Experiment {k+1}\n  Time: {time.time()-st}\n  {m}")

total_score = score/num_exp
total_scores = scores/num_exp
m = f'\n  '.join([c.capitalize() + ": " + str(float(total_score[i])) for i, c in enumerate(opt['metric'])])
print(f"Experiment Summary\n  Time: {time.time()-tt}\n  Number of Experiments: {num_exp}\n  {m}")

for i in range(5):
    plt.plot(np.arange(opt['epoch']), total_scores[:, i])
plt.legend(opt['metric'])
plt.savefig(f'output/{args.dataset}.png')
