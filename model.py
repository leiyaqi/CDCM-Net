""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

import math
import copy


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


    def forward(self, data_list):

        confidence_loss = []

        FeatureInfo, feature, Logit, Confidence = dict(), dict(), dict(), dict()
        for view in range(self.views):

            feature[view] = data_list[view]

            Logit[view] = nn.Softmax(self.relu(self.ClassifierLayer[view](feature[view])))
            Confidence[view] = self.relu2(self.ConfidenceLayer[view](feature[view]))


        return  Confidence,Logit



class MMDF(nn.Module):
    def __init__(self):
        super(MMDF, self).__init__()
        self.img_encoder = TransformerEncoder(True, 3)

        self.text_encoder = TransformerEncoder(True, 3)

        self.fusion_encoder = TransformerEncoder(True, 6)

        self.Confidence = MMD(in_dim=[512, 512], dropout=0.5)
        self.proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 10),
            nn.ReLU())

    def forward(self, img_feature, text_feature, label):
        conf_img_feature, _ = self.img_encoder(img_feature)
        conf_text_feature, _ = self.text_encoder(text_feature)


        mmfeature = []

        mmfeature.append(conf_img_feature[:, 0, :])
        mmfeature.append(conf_text_feature[:, 0, :])

        weight, Logit = self.Confidence(mmfeature, label)
        # print(weight[0]-weight[1])
        img_feature = img_feature.reshape(-1, 25600) * (weight[0])
        img_feature = img_feature.reshape(-1, 50, 512)
        text_feature = text_feature.reshape(-1, 25600) * (weight[1])
        text_feature = text_feature.reshape(-1, 50, 512)
        cross_feature = torch.cat((img_feature, text_feature), dim=1).type(torch.float)

        cross_feature, _ = self.fusion_encoder(cross_feature)

        distribution = self.proj(torch.cat((cross_feature[:, 0, :], cross_feature[:, 50, :]), dim=1))
        distribution = nn.Softmax(distribution)
        return distribution


class TransformerEncoder(nn.Module):
    def __init__(self, vis, deep):
        super(TransformerEncoder, self).__init__()
        self.vis = vis
        hidden_size = 512
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(deep):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, ):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = 12
        self.attention_head_size = 7
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(512, self.all_head_size)
        self.key = nn.Linear(512, self.all_head_size)
        self.value = nn.Linear(512, self.all_head_size)
        self.dropout = nn.Dropout(0.5)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)



        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores
        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = torch.reshape(context_layer, new_context_layer_shape)
        return context_layer


# 2.构建self-Attention模块
class Attention(nn.Module):
    def __init__(self, hidden_size=512, vis=True):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 16  # 12
        self.attention_head_size = int(hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)  # wm,768->768
        self.attn_dropout = nn.Dropout()
        self.proj_dropout = nn.Dropout()

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,64)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)  # wm,768->768
        mixed_key_layer = self.key(hidden_states)  # wm,768->768
        mixed_value_layer = self.value(hidden_states)  # wm,768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)  # wm，(bs,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights  # wm,(bs,197,768),(bs,197,197)


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x





class Block(nn.Module):
    def __init__(self, hidden_size=512, vis=True):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Memory(nn.Module):

    def __init__(self, radius=16.0, n_slot=512):
        super().__init__()

        self.key = nn.Parameter(torch.Tensor(n_slot, 512), requires_grad=True)
        nn.init.normal_(self.key, 0, 0.5)
        self.value = nn.Parameter(torch.Tensor(n_slot, 512), requires_grad=True)
        nn.init.normal_(self.value, 0, 0.5)

        self.q_embd = nn.Linear(512, 512)
        self.v_embd = nn.Linear(512, 512)

        self.fusion = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.5)

        self.radius = radius
        self.softmax = nn.Softmax(1)

    def forward(self, query, value=None, inference=False):
        # B, S, 512
        B, S, C = query.size()
        mer_query = query.view(B * S, -1)
        add_loss, tr_fusion, recon_loss = None, None, None

        key_norm = F.normalize(self.key, dim=1)
        embd_query = self.q_embd(mer_query)
        key_sim = F.linear(F.normalize(embd_query, dim=1), key_norm)
        key_add = self.softmax(self.radius * key_sim)

        vir_aud = torch.matmul(key_add, self.value.detach())

        te_fusion = torch.cat([query, vir_aud.view(B, S, -1)], 2)
        te_fusion = self.dropout(te_fusion)
        te_fusion = self.fusion(te_fusion)

        # Loss gen
        if not inference:
            mer_value = value.view(B * S, -1)
            embd_value = self.v_embd(mer_value.detach())
            value_norm = F.normalize(self.value, dim=1)
            value_sim = F.linear(F.normalize(embd_value, dim=1), value_norm)
            value_add = self.softmax(self.radius * value_sim)

            aud = torch.matmul(value_add, self.value)

            recon_loss = F.mse_loss(aud, mer_value.detach())
            recon_loss = recon_loss.unsqueeze(0)


            add_loss = F.kl_div(torch.log(key_add), value_add.detach(), reduction='batchmean')
            add_loss = add_loss.unsqueeze(0)

        return te_fusion, recon_loss, add_loss