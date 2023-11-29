import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

# TSNE 


def Draw(epoch, V_list, R_list, I_list):
    IR_List = torch.cat((I_list, R_list), dim=0)
    VR_List = torch.cat((V_list, R_list), dim=0)
    VR_tsne = torch.tensor(TSNE(n_components=2).fit_transform(VR_List.cpu()))
    IR_tsne = torch.tensor(TSNE(n_components=2).fit_transform(IR_List.cpu()))
    VR_label = torch.cat((torch.ones(V_list.shape[0]), torch.zeros(R_list.shape[0])), dim=0)
    IR_label = torch.cat((torch.ones(I_list.shape[0]), torch.zeros(R_list.shape[0])), dim=0)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(str(epoch) + "IR_tsne")
    plt.scatter(IR_tsne[:, 0], IR_tsne[:, 1], c=IR_label)
    plt.subplot(1, 2, 2)
    plt.title(str(epoch) + "VR_tsne")
    plt.scatter(VR_tsne[:, 0], VR_tsne[:, 1], c=VR_label)
    plt.subplot(122)
    plt.colorbar()
    plt.draw()
    plt.savefig('./img/photonet-{}.png'.format(epoch + 1))
    plt.pause(1)
    plt.close()
