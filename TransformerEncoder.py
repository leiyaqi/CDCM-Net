import math
import torch
import torch.nn.functional as F
from torch import nn
from function import distribution_normalize
import copy
from MFB import Fusion
from MMD import MMD

Is_AVA = True
if Is_AVA:
    Distribution = 10
else:
    Distribution = 7

class MMDF(nn.Module):
    def __init__(self):
        super(MMDF, self).__init__()
        self.img_encoder = TransformerEncoder(True, 3)

        self.text_encoder = TransformerEncoder(True, 3)

        self.fusion_encoder = TransformerEncoder(True, 6)

        self.Confidence = MMD(in_dim=[512, 512], dropout=0.5)
        self.proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, Distribution),
            nn.ReLU())

    def forward(self, img_feature, text_feature, label):
        # 
        conf_img_feature, _ = self.img_encoder(img_feature)
        conf_text_feature, _ = self.text_encoder(text_feature)
        # conf_img_feature = img_feature
        # conf_text_feature = text_feature

        mmfeature = []

        mmfeature.append(conf_img_feature[:, 0, :])
        mmfeature.append(conf_text_feature[:, 0, :])

        ConfidenceLoss, weight, Logit = self.Confidence(mmfeature, label)
        # print(weight[0]-weight[1])
        img_feature = img_feature.reshape(-1, 25600) * (weight[0])
        img_feature = img_feature.reshape(-1, 50, 512)
        text_feature = text_feature.reshape(-1, 25600) * (weight[1])
        text_feature = text_feature.reshape(-1, 50, 512)
        cross_feature = torch.cat((img_feature, text_feature), dim=1).type(torch.float)

        # 
        cross_feature, _ = self.fusion_encoder(cross_feature)

        distribution = self.proj(torch.cat((cross_feature[:, 0, :], cross_feature[:, 50, :]), dim=1))
        distribution = distribution_normalize(distribution)
        return distribution, ConfidenceLoss, weight, Logit




# 2.MFB
class MFBFusion(nn.Module):
    def __init__(self):
        super(MFBFusion, self).__init__()

        self.cross_fc = Fusion()

    def forward(self, img_feature, text_feature):
        # cross_feature = torch.cat((img_feature[:, 0], text_feature[:, 0]), dim=1)

        output = self.cross_fc(img_feature[:, 0], text_feature[:, 0])

        output = distribution_normalize(output)
        return output,None


# 3.concat 
class CCFusion(nn.Module):
    def __init__(self):
        super(CCFusion, self).__init__()
        # self.encoder = TransformerEncoder(True)
        self.cross_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, Distribution),
            nn.ReLU())

    def forward(self, img_feature, text_feature,GroundTruth):
        # cross_feature = torch.cat((img_feature, text_feature), dim=1).type(torch.float)
        # cross_feature, weights = self.encoder(cross_feature)
        # cross_feature = torch.cat((cross_feature[:, 0], cross_feature[:, 50]), dim=1)
        cross_feature = torch.cat((img_feature[:, 0], text_feature[:, 0]), dim=1).type(torch.float)
        output = self.cross_fc(cross_feature)

        output = distribution_normalize(output)
        return output,None


# 5.transformer img


class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        self.img_encoder = TransformerEncoder(True, 6)

        self.proj = nn.Sequential(
            nn.Linear(512, Distribution),
            nn.ReLU())

    def forward(self,img_feature, text_feature,GroundTruth):
        img_feature, _ = self.img_encoder(img_feature)
        # conf_text_feature, _ = self.text_encoder(text_feature)
        # conf_img_feature = img_feature
        # conf_text_feature = text_feature
        #
        # mmfeature = []
        #
        # mmfeature.append(conf_img_feature[:, 0, :])
        # mmfeature.append(conf_text_feature[:, 0, :])
        #
        # ConfidenceLoss, weight = self.Confidence(mmfeature, label)
        # img_feature = img_feature.reshape(-1, 25600) * (weight[0] + 1)
        # img_feature = img_feature.reshape(-1, 50, 512)
        # text_feature = text_feature.reshape(-1, 25600) * (weight[1] + 1)
        # text_feature = text_feature.reshape(-1, 50, 512)
        # cross_feature = torch.cat((img_feature, text_feature), dim=1).type(torch.float)

        # 
        # cross_feature, _ = self.fusion_encoder(cross_feature)

        distribution = self.proj(img_feature[:, 0, :])
        distribution = distribution_normalize(distribution)
        return distribution, None



# 

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

        # attention_score

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

        # context
        # layerattn_probs
        # 
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = torch.reshape(context_layer, new_context_layer_shape)
        return context_layer


# 2.self-Attention
class Attention(nn.Module):
    def __init__(self, hidden_size=512, vis=True):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 16  # 12
        self.attention_head_size = int(hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = nn.Linear(hidden_size, self.all_head_size)  # wm,768->768，Wq（768,768）
        self.key = nn.Linear(hidden_size, self.all_head_size)  # wm,768->768,Wk（768,768）
        self.value = nn.Linear(hidden_size, self.all_head_size)  # wm,768->768,Wv（768,768）
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
        # hidden_states：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)  # wm,768->768
        mixed_key_layer = self.key(hidden_states)  # wm,768->768
        mixed_value_layer = self.value(hidden_states)  # wm,768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)  # wm，(bs,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # （bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 
        attention_probs = self.softmax(attention_scores)  # softmax,
        weights = attention_probs if self.vis else None  # wm,
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  #
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights  # wm,(bs,197,768),(bs,197,197)


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(512, 1024)  # wm,786->3072
        self.fc2 = nn.Linear(1024, 512)  # wm,3072->786
        self.act_fn = torch.nn.functional.gelu  # wm
        self.dropout = nn.Dropout()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  # wm,786->3072
        x = self.act_fn(x)  
        x = self.dropout(x)  # wm
        x = self.fc2(x)  # wm3072->786
        x = self.dropout(x)
        return x



class Block(nn.Module):
    def __init__(self, hidden_size=512, vis=True):
        super(Block, self).__init__()
        self.hidden_size = hidden_size  # wm,768
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)  # wm，
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h  # 

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h  # 
        return x, weights


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
