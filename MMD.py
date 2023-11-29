""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from function import EMDLoss2,distribution_normalize,distribution_to_total_score,EMDLoss1


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        # x = self.relu(x)
        return x

class MMD(nn.Module):
    def __init__(self, in_dim, dropout):
        super(MMD,self).__init__()
        hidden_dim = [512]
        self.views = len(in_dim)
        self.dropout = dropout
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.ConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.ClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 10) for _ in range(self.views)])
        # self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        # self.MMClasifier = []
        # for layer in range(1, len(hidden_dim) - 1):
        #     self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
        #
        #     self.MMClasifier.append(nn.Dropout(p=dropout))
        # if len(self.MMClasifier):
        #     self.MMClasifier.append(LinearLayer(hidden_dim[-1], 10))
        #     self.MMClasifier.append(nn.ReLU())
        # else:
        #     self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], 10))
        #     self.MMClasifier.append(nn.ReLU())
        # self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, label):

        confidence_loss = []
        criterion1 = EMDLoss1()
        criterion2 = EMDLoss2()
        FeatureInfo, feature, Logit, Confidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            # FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](data_list[view]))
            # feature[view] = data_list[view] * FeatureInfo[view]
            feature[view] = data_list[view]
            # feature[view] = self.FeatureEncoder[view](feature[view])
            # feature[view] = F.relu(feature[view])
            # feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            Logit[view] = distribution_normalize(self.relu(self.ClassifierLayer[view](feature[view])))
            Confidence[view] = self.relu2(self.ConfidenceLayer[view](feature[view]))
            p_target2 = criterion2(Logit[view], label)
            p_target1 = criterion1(Logit[view],label)
            confidence_loss.append(F.mse_loss(Confidence[view].view(-1), 1.0 - p_target1) + p_target2)

        return confidence_loss, Confidence,Logit




