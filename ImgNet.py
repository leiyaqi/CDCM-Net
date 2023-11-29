from PIL import Image
from TransformerEncoder import TransformerEncoder
import warnings
from torch import nn
import torch
import clip

warnings.filterwarnings('ignore')




class Img_net(nn.Module):
    def __init__(self):
        super(Img_net, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda", jit=False)
        self.encoder = TransformerEncoder(True)
        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 10),
                                nn.ReLU())

    def forward(self, number_idx, text_token, inference):


        img_list = []
        for number in number_idx:
            #
            with torch.no_grad():
                # file_path = "/media/zxd/1/zx_codes/Databases/AVA/images/" + number + ".jpg"
                file_path = "/media/zxd/databases/AVA/images/" + number + ".jpg"
                img = self.preprocess(Image.open(file_path).convert('RGB')).unsqueeze(0).to("cuda")
                img = self.model.encode_image(img)
                # x = img[:, 0, :]
                x = img.type(torch.float)
                img = x.unsqueeze(0)
                img_list.append(img)
        img = torch.stack(img_list)
        image_feature = torch.squeeze(img)
        image_feature, weights = self.encoder(image_feature)
        image_feature = image_feature[:, 0, :]
        x = self.fc(image_feature)

        return x
