import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

Is_AVA = False
if Is_AVA:
    Distribution = 10
else:
    Distribution = 7
class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.Linear_t_proj = nn.Linear(512, 2560)
        self.Linear_i_proj = nn.Linear(512, 2560)
        self.Linear_predict = nn.Linear(512, Distribution)
        # self.Linear_predict = nn.Linear(512, 7)
        self.Dropout_M = nn.Dropout(0.3)

    def forward(self, input_text,input_image):

        mfb_q_proj = self.Linear_t_proj(input_image)  # torch.Size([5, 2560])
        #print(mfb_q_proj.shape)
        mfb_i_proj = self.Linear_i_proj(input_text)  # torch.Size([5, 2560])
        #print(mfb_i_proj.shape)

        mfb_iq_eltwise = torch.mul(mfb_q_proj, mfb_i_proj)  # torch.Size([5, 2560])
        #print(mfb_iq_eltwise.shape)

        mfb_iq_drop = self.Dropout_M(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(input_image.shape[0], 1, 512, 5)


        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)  # torch.Size([5, 1, 512, 1])

        mfb_out = torch.squeeze(mfb_iq_sumpool)  # torch.Size([5,512])
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))  #Power Normalization
        # print(mfb_sign_sqrt.shape) # torch.Size([5, 512])
        if len(mfb_sign_sqrt.shape) == 1:
            mfb_sign_sqrt = mfb_sign_sqrt.unsqueeze(0)
        mfb_l2 = F.normalize(mfb_sign_sqrt) #L2 Normalization
        # print(mfb_l2.shape)
        prediction = self.Linear_predict(mfb_l2)#torch.Size([2,10])
        #print(prediction.shape)
        # prediction = F.softmax(prediction)

        return prediction


class MixtureModel_MFB_2(nn.Module):
    def __init__(self, args):
        super(MixtureModel_MFB_2, self).__init__()
        self.args = args


        self.fusion_model = Fusion()
        self.dim_reduction = nn.Sequential(
            nn.Linear(1 * 2048, 768)
        )
        self.softmax = nn.Softmax()



    def forward(self, img_feature,text_feature):
        # img_feature = self.visual_model(img) # torch.Size([2, 64, 2048])  torch.Size([5, 49, 2048])
        #print(img_feature.shape) #torch.Size([5, 49, 2048])
        # img_feature = img_feature.reshape(img_feature.size(0), -1)  # torch.Size([2, 131072])
        #print(img_feature.shape) #torch.Size([5, 100352])
        # img_feature = self.dim_reduction(img_feature)#torch.Size([5, 512])
        #print(img_feature.shape) #torch.Size([5, 768])


        # text_feature = self.text_model(txt, mask, segment)
        #print(text_feature.shape) #torch.Size([5, 768])

        prediction = self.fusion_model(img_feature, text_feature)  #  torch.Size([5, 10])


        prediction = self.softmax(prediction)
        #print(prediction.shape) #torch.Size([5, 10])


        return prediction


if __name__ == '__main__':
    model = Fusion()
    a = torch.rand((4,512))
    b = torch.rand((4,512))
    out = model(a,b)
    # b = F.normalize(a)  # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
    print(out.shape)