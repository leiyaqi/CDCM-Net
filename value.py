from model import *
from PIL import Image
import warnings
from torch import nn
import torch
import clip

warnings.filterwarnings('ignore')


class MMencoder(nn.Module):
    def __init__(self):
        super(MMencoder, self).__init__()

        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda", jit=False)

    def forward(self, file_paths):
        img_list = []
        for file_path in file_paths:
            img = self.preprocess(Image.open(file_path).convert('RGB')).unsqueeze(0).to("cuda")
            img = self.model.encode_image(img).detach()
            img = img.unsqueeze(0)
            img_list.append(img)
        img = torch.stack(img_list)
        image_feature = torch.squeeze(img).type(torch.float)

        return image_feature


mmencoder = MMencoder().cuda()
mmfusion = MMDF().cuda()
memory = Memory().cuda()
img_feature = mmencoder(["imgpath"])
fake_text, reloss, add_loss = memory(img_feature, None, True)
distribution = mmfusion(img_feature, fake_text, None)
print(distribution)