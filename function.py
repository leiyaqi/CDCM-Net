import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable
from torch import nn




def mean_score(scores):
    """ calculate mean score for AVA dataset
    :param scores:
    :return: row wise mean score if scores contains multiple rows, else
             a single mean score
    """
    si = np.arange(1, 11, 1).reshape(10, 1)
    mean = np.dot(scores, si)
    return mean


def distribution_normalize(distribution):
  
    norm_distribution = torch.ones_like(distribution)

    if len(distribution.shape) is 1:
        if torch.mean(distribution) is 0.0:
            return torch.zeros_like(distribution)
        total_num = sum(distribution)
        for j in range(len(distribution)):
            norm_distribution[j] = distribution[j] / total_num
    else:
        I, J = distribution.shape
        for i in range(I):
            if torch.mean(distribution[i]) is 0.0:
                distribution[i] = torch.zeros_like(distribution)
                continue
            total_num = sum(distribution[i])
            for j in range(J):
                norm_distribution[i][j] = distribution[i][j] / total_num

    return norm_distribution


def accuracy(y_test, y_pred):
    accuracy_final = accuracy_score(y_test, y_pred)
    return accuracy_final


def score_to_grade1(score, delta=0):
    """
    if delta == 0:
        1.0 -- 5.0: bad 0
        5.0 -- 10.0: good 1
        else -1
    if delta != 0:
        1.0 -- 5.0-delta: bad 0
        5.0+-delta: middle 1
        5.0+delta -- 10.0: good 2
        else -1
    """
    mid = 5
    if delta == 0:
        if score < mid:
            return 0
        elif score >= mid:
            return 1
        else:
            return -1
    else:
        if score <= mid - delta:
            return 0
        elif mid - delta < score < mid + delta:
            return 1
        elif mid + delta <= score <= 10.0:
            return 2
        else:
            return -1


def distribution_to_total_score(distribution):
    """
    distribution score to total score
    :param distribution: a list, aesthetic distribution score of an image
    :return: total score
    """
    s = 0
    for k in range(len(distribution)):  # 10
        s += (k + 1) * distribution[k]  # score num from 1 to 10
    total = s / sum(distribution)
    return total


# EMD LOSS
class EMDLoss1(nn.Module):
    def __init__(self):
        super(EMDLoss1, self).__init__()

    def forward(self, p_target: Variable, p_estimate: Variable):
        # p_estimate = F.softmax(p_estimate, dim=1)  # todo softmax loss
        # p_target = F.softmax(p_target,dim=1)
        # print(p_target.shape)  # torch.Size([6, 7])
        # print(p_estimate.shape)  # torch.Size([6, 7])
        # print(p_target)  # tensor([[0.0000, 0.0000, 0.1000, 0.2000, 0.7000, 0.0000, 0.0000], ...
        # print(p_estimate)  # tensor([[0.0810, 0.1458, 0.1313, 0.0993, 0.1991, 0.1916, 0.1519], ...

        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.mean(torch.abs(cdf_diff))
        return samplewise_emd.mean()


class EMDLoss2(nn.Module):
    def __init__(self):
        super(EMDLoss2, self).__init__()

    def forward(self, p_target: Variable, p_estimate: Variable):
        # p_estimate = F.softmax(p_estimate, dim=1)  # todo  softmax  loss
        # p_target = F.softmax(p_target,dim=1)
        # print(p_target.shape)  # torch.Size([6, 7])
        # print(p_estimate.shape)  # torch.Size([6, 7])
        # print(p_target)  # tensor([[0.0000, 0.0000, 0.1000, 0.2000, 0.7000, 0.0000, 0.0000], ...
        # print(p_estimate)  # tensor([[0.0810, 0.1458, 0.1313, 0.0993, 0.1991, 0.1916, 0.1519], ...

        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()
