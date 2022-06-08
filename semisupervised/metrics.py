import torch
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from torch.nn.functional import softmax

DEVICE = 'cuda:0'


def accuracy(label, preds):
    pred = torch.max(preds, dim=1)[-1]  #  + torch.ones(label.shape[0], device=DEVICE)
    return (label == pred).int().sum()/label.shape[0]


def precision(label, preds):
    # return average_precision_score(label.cpu(), preds.cpu())
    pred = torch.max(preds, dim=1)[-1]  #  + torch.ones(label.shape[0], device=DEVICE)

    classes = list(set(torch.hstack((label, pred)).tolist()))
    true = (label == pred).int()
    false = (label != pred).int()
    n = len(classes)
    scores = 0
    for c in classes:
        tp = (true * (pred == c).int()).sum()
        fp = (false * (false == c).int()).sum()
        if tp+fp == 0:
            n -= 1
            continue
        scores += tp/(tp+fp)
    return scores/n


def recall(label, preds):
    # return recall_score(label.cpu(), preds.cpu())
    pred = torch.max(preds, dim=1)[-1]  #  + torch.ones(label.shape[0], device=DEVICE)
    classes = list(set(torch.hstack((label, pred)).tolist()))
    true = (label == pred).int()
    false = (label != pred).int()
    n = len(classes)
    scores = 0
    for c in classes:
        tp = (true * (pred == c).int()).sum()
        fn = (false * (true == c).int()).sum()
        if tp+fn == 0:
            n -= 1
            continue
        scores += tp/(tp+fn)
    return scores/n


def f1(label, preds):
    p = precision(label, preds)
    r = recall(label, preds)
    return 2*p*r/(p+r)


def auroc(label, preds):
    return torch.tensor(roc_auc_score(label.cpu(), preds.cpu(), multi_class='ovo'))


if __name__ == "__main__":
    DEVICE = 'cpu'
    label = torch.tensor([0, 1, 2, 3], device=DEVICE)
    preds = torch.vstack([torch.rand(4, device=DEVICE) for i in range(4)]) + \
            torch.tensor([[1., 0., 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.],
                          [0., 0., 0., 1.]], device=DEVICE)
    preds = softmax(preds)
    mMap = {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc}
    metric = ["accuracy", "precision", "recall", "f1", "auroc"]
    result = []
    for m in metric:
        metric = mMap[m]
        result.append(metric(label, preds).reshape(1))
    print(result)
    result = torch.cat(result)
    print(result)
