from PIL import Image
import warnings
from torch import nn
import torch
import clip

warnings.filterwarnings('ignore')


# 


class MMencoder(nn.Module):
    def __init__(self):
        super(MMencoder, self).__init__()

        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda", jit=False)

    def forward(self, number_idx, text_token):
        img_list = []
        text_list = []
       

        for number in number_idx:
        
            # file_path = "/media/zxd/1/zx_codes/Databases/AVA/images/" + number + ".jpg"
            file_path = "/media/zxd/databases/AVA/images/" + number + ".jpg"
            # file_path = "/media/zxd/databases/photonet/photos/" + number + "-md.jpg"
            img = self.preprocess(Image.open(file_path).convert('RGB')).unsqueeze(0).to("cuda")
            img = self.model.encode_image(img).detach()
            img = img.unsqueeze(0)
            img_list.append(img)
        img = torch.stack(img_list)
        image_feature = torch.squeeze(img).type(torch.float)
        for item in text_token:
            text_feature = self.model.encode_text(item.to("cuda")).type(torch.float).detach()
            text_feature = text_feature.unsqueeze(0)
            if text_feature.shape[1] < 50:
                expand = torch.zeros(1, 50 - text_feature.shape[1], 512).to("cuda")
                text_feature = torch.cat((text_feature, expand), dim=1)
            else:
                text_feature = text_feature[:, :50, :]
            text_list.append(text_feature)
        text_feature = torch.stack(text_list)
        text_feature = torch.squeeze(text_feature)

        return image_feature, text_feature
