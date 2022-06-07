import copy
import numpy as np
import random
import torch

from trainer import Trainer
from gnn import GNNq, GNNp
import loader


class Train:
    def __init__(self, opt):

        self.opt = opt

        torch.manual_seed(opt["seed"])
        np.random.seed(opt["seed"])
        random.seed(opt["seed"])
        if opt["cpu"]:
            cuda = False
        elif opt["cuda"]:
            torch.cuda.manual_seed(opt["seed"])


        net_file = self.opt['data'] + '/net.txt'
        label_file = self.opt['data'] + '/label.txt'
        feature_file = self.opt['data'] + '/feature.txt'
        train_file = self.opt['data'] + '/train.txt'
        dev_file = self.opt['data'] + '/dev.txt'
        test_file = self.opt['data'] + '/test.txt'

        vocab_node = loader.Vocab(net_file, [0, 1])
        vocab_label = loader.Vocab(label_file, [1])
        vocab_feature = loader.Vocab(feature_file, [1])

        self.opt['num_node'] = len(vocab_node)
        self.opt['num_feature'] = len(vocab_feature)
        self.opt['num_class'] = len(vocab_label)

        graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
        label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
        feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
        graph.to_symmetric(self.opt['self_link_weight'])
        feature.to_one_hot(binary=True)
        self.adj = graph.get_sparse_adjacency(self.opt['cuda'])

        with open(train_file, 'r') as fi:
            idx_train = [vocab_node.stoi[line.strip()] for line in fi]
        with open(dev_file, 'r') as fi:
            idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
        with open(test_file, 'r') as fi:
            idx_test = [vocab_node.stoi[line.strip()] for line in fi]
        idx_all = list(range(self.opt['num_node']))

        self.inputs = torch.Tensor(feature.one_hot)
        self.target = torch.LongTensor(label.itol)
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_dev = torch.LongTensor(idx_dev)
        self.idx_test = torch.LongTensor(idx_test)
        self.idx_all = torch.LongTensor(idx_all)
        self.inputs_q = torch.zeros(self.opt['num_node'], self.opt['num_feature'])
        self.target_q = torch.zeros(self.opt['num_node'], self.opt['num_class'])
        self.inputs_p = torch.zeros(self.opt['num_node'], self.opt['num_class'])
        self.target_p = torch.zeros(self.opt['num_node'], self.opt['num_class'])

        if self.opt['cuda']:
            self.inputs = self.inputs.cuda()
            self.target = self.target.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_dev = self.idx_dev.cuda()
            self.idx_test = self.idx_test.cuda()
            self.idx_all = self.idx_all.cuda()
            self.inputs_q = self.inputs_q.cuda()
            self.target_q = self.target_q.cuda()
            self.inputs_p = self.inputs_p.cuda()
            self.target_p = self.target_p.cuda()

        gnnq = GNNq(self.opt, self.adj)
        self.trainer_q = Trainer(self.opt, gnnq)

        gnnp = GNNp(self.opt, self.adj)
        self.trainer_p = Trainer(self.opt, gnnp)

    def train(self):
        base_results, q_results, p_results = [], [], []
        base_results += self.pre_train(self.opt['pre_epoch'])
        for k in range(self.opt['iter']):
            p_results += self.train_p(self.opt['epoch'])
            q_results += self.train_q(self.opt['epoch'])

        acc_test = self.get_accuracy(q_results)

        print('{:.3f}'.format(acc_test * 100))

        if self.opt['save'] != '/':
            self.trainer_q.save(self.opt['save'] + '/gnnq.pt')
            self.trainer_p.save(self.opt['save'] + '/gnnp.pt')

    def init_q_data(self):
        self.inputs_q.copy_(self.inputs)
        temp = torch.zeros(self.idx_train.size(0), self.target_q.size(1)).type_as(self.target_q)
        temp.scatter_(1, torch.unsqueeze(self.target[self.idx_train], 1), 1.0)
        self.target_q[self.idx_train] = temp

    def update_p_data(self):
        preds = self.trainer_q.predict(self.inputs_q, self.opt['tau'])
        if self.opt['draw'] == 'exp':
            self.inputs_p.copy_(preds)
            self.target_p.copy_(preds)
        elif self.opt['draw'] == 'max':
            idx_lb = torch.max(preds, dim=-1)[1]
            self.inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            self.target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        elif self.opt['draw'] == 'smp':
            idx_lb = torch.multinomial(preds, 1).squeeze(1)
            self.inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
            self.target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        if self.opt['use_gold'] == 1:
            temp = torch.zeros(self.idx_train.size(0), self.target_q.size(1)).type_as(self.target_q)
            temp.scatter_(1, torch.unsqueeze(self.target[self.idx_train], 1), 1.0)
            self.inputs_p[self.idx_train] = temp
            self.target_p[self.idx_train] = temp

    def update_q_data(self):
        preds = self.trainer_p.predict(self.inputs_p)
        self.target_q.copy_(preds)
        if self.opt['use_gold'] == 1:
            temp = torch.zeros(self.idx_train.size(0), self.target_q.size(1)).type_as(self.target_q)
            temp.scatter_(1, torch.unsqueeze(self.target[self.idx_train], 1), 1.0)
            self.target_q[self.idx_train] = temp

    def pre_train(self, epoches):
        best = 0.0
        self.init_q_data()
        results = []
        for epoch in range(epoches):
            loss = self.trainer_q.update_soft(self.inputs_q, self.target_q, self.idx_train)
            _, preds, accuracy_dev = self.trainer_q.evaluate(self.inputs_q, self.target, self.idx_dev)
            _, preds, accuracy_test = self.trainer_q.evaluate(self.inputs_q, self.target, self.idx_test)
            results += [(accuracy_dev, accuracy_test)]
            if accuracy_dev > best:
                best = accuracy_dev
                state = dict([('model', copy.deepcopy(self.trainer_q.model.state_dict())), ('optim', copy.deepcopy(self.trainer_q.optimizer.state_dict()))])
        self.trainer_q.model.load_state_dict(state['model'])
        self.trainer_q.optimizer.load_state_dict(state['optim'])
        return results

    def train_p(self, epoches):
        self.update_p_data()
        results = []
        for epoch in range(epoches):
            loss = self.trainer_p.update_soft(self.inputs_p, self.target_p, self.idx_all)
            _, preds, accuracy_dev = self.trainer_p.evaluate(self.inputs_p, self.target, self.idx_dev)
            _, preds, accuracy_test = self.trainer_p.evaluate(self.inputs_p, self.target, self.idx_test)
            results += [(accuracy_dev, accuracy_test)]
        return results

    def train_q(self, epoches):
        self.update_q_data()
        results = []
        for epoch in range(epoches):
            loss = self.trainer_q.update_soft(self.inputs_q, self.target_q, self.idx_all)
            _, preds, accuracy_dev = self.trainer_q.evaluate(self.inputs_q, self.target, self.idx_dev)
            _, preds, accuracy_test = self.trainer_q.evaluate(self.inputs_q, self.target, self.idx_test)
            results += [(accuracy_dev, accuracy_test)]
        return results

    def get_accuracy(self, results):
        best_dev, acc_test = 0.0, 0.0
        for d, t in results:
            if d > best_dev:
                best_dev, acc_test = d, t
        return acc_test