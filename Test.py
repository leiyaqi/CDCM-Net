import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import pearsonr
# from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from function import *
from Draw import Draw
import pandas as pd


Is_AVA =True

def MyTest(epoch, mmencoder,memory,mmfusion, test_loader,ismemroy):
    #EMD loss λ=1
    criterion = EMDLoss1()
    batch_item = 0
    mmfusion.eval()
    mmencoder.eval()
    if ismemroy:
        memory.eval()
    Test_totalloss = 0.0
    Avescore_list = []
    Label_list = []
    data_num = 0
    total = 0

    EMD_Total_loss = 0.0
    conf_total = 0.0
    re_total_loss = 0.0
    COS_distance = 0.0
    V_list = None
    R_list = None
    I_list = None
    if Is_AVA:
        Distribution = int(10)
    else:
        Distribution = int(7)
    with torch.no_grad():
        result_list = []
        groundtruth_list = []
        ID_list = []
        confidence_list1 = []
        confidence_list2 = []
        Logit1 = []
        Logit2 = []
        distribution_list = []
        GroundTruth_list = []
        for i, data in enumerate(tqdm(test_loader)):
            number_idx = data[0]
            text_L = data[1]
            batch_item = i
            GroundTruth = torch.tensor(data[2])
            GroundTruth = GroundTruth.reshape(-1, Distribution)
            GroundTruth = distribution_normalize(torch.tensor(GroundTruth, dtype=torch.float))
            GroundTruth = torch.tensor(GroundTruth).to('cuda')

            data_num += GroundTruth.shape[0]

            # 

            img_feature, text_feautre = mmencoder(number_idx, text_L)

            # Memory faketext text
            if ismemroy:
                fake_text,  reloss, add_loss = memory(img_feature, text_feautre, False)
                distribution, ConfidenceLoss,WEIGHT,logit= mmfusion(img_feature, fake_text,GroundTruth)
            else:
                distribution, ConfidenceLoss,WEIGHT,logit = mmfusion(img_feature, text_feautre,GroundTruth)
            distribution = torch.squeeze(distribution, dim=1)
            for view in  range(len(ConfidenceLoss)):
                conf_total += ConfidenceLoss[view].item()

            distribution = distribution.reshape(-1, Distribution)
            # EMD loss
            emd_loss = criterion(distribution, GroundTruth)
            if ismemroy:
                EMD_Total_loss +=emd_loss.item()
                re_total_loss += reloss.item() + add_loss.item()
                total_loss = emd_loss + reloss + add_loss
            else:
                EMD_Total_loss+=emd_loss.item()
                total_loss = emd_loss
            Test_totalloss += total_loss.item()

            for num in range(len(GroundTruth)):
                # 
                distribution_list.append(distribution[num].cpu().numpy())#6.distribution
                R_s = distribution_to_total_score(distribution[num].cpu())
                Gr_s = distribution_to_total_score(GroundTruth[num].cpu())
                Label_list.append(Gr_s.item())#9.GroundTruth score
                Avescore_list.append(R_s.item())#7.predict score
                GroundTruth_list.append(GroundTruth[num].cpu().numpy())#8.GroundTruth
                ID_list.append(number_idx[num])#1.ID
                confidence_list1.append(WEIGHT[0][num].item())#2.conf1
                confidence_list2.append(WEIGHT[1][num].item())#3.conf2
                Logit1.append(logit[0][num].cpu().numpy())#4.logit1
                Logit2.append(logit[1][num].cpu().numpy())#5.logit2
                #
                R_tar = score_to_grade1(R_s)
                Gr_tar = score_to_grade1(Gr_s)
                if R_tar is -1 or Gr_tar is -1:
                    print("ERROR!")
                    return -1
                # 
                result_list.append(R_tar)#10.predict grade
                groundtruth_list.append(Gr_tar)#11.GroundTruth grade

        #     if ismemroy:
        #         # TSNE
        #         total += GroundTruth.shape[0]
        #
        #         fake_text = fake_text.unsqueeze(dim=0)
        #         text_feat = text_feautre.unsqueeze(dim=0)
        #         img_feat = img_feature
        #         bz = len(GroundTruth)
        #         #
        #         fake_sample = F.interpolate(fake_text, size=[1,512]).squeeze().reshape(bz, -1).type(torch.float16)
        #         real_sample = F.interpolate(text_feat, size=[1,512]).squeeze().reshape(bz, -1).type(torch.float16)
        #         for v_num in range(bz):
        #             fake_sample = F.normalize(fake_sample)
        #             real_sample = F.normalize(real_sample)
        #             img_sample = F.normalize(img_feat[:,0,:].reshape(bz, -1)).type(torch.float16)
        #             if V_list is None:
        #                 V_list = fake_sample
        #                 R_list = real_sample
        #                 I_list = img_sample
        #             else:
        #                 V_list = torch.cat((V_list, fake_sample), dim=0)
        #                 R_list = torch.cat((R_list, real_sample), dim=0)
        #                 I_list = torch.cat((I_list, img_sample), dim=0)
        #         # 
        # chat = list(zip(ID_list,confidence_list1,confidence_list2,Logit1,Logit2,distribution_list,Avescore_list,GroundTruth_list,
        #                 Label_list,result_list,groundtruth_list))
        # torch.save(chat,"data_chat.pth")


        # name_attribute=['ID','conf1','conf2','logit1','logit2','distribution','predictS','groundtruth','groundtruthS','predictG','groundtruthG']
        # writerCSV = pd.DataFrame(columns=name_attribute, data=chat)
        # writerCSV.to_csv('data_chat.csv', encoding='utf-8')
        # if ismemroy:
        #     I_list = I_list[:500].cpu()
        #     R_list = R_list[:500].cpu()
        #     V_list = V_list[:500].cpu()
        #     # Draw(epoch, V_list, R_list, I_list)

        # tensor
        Avescore_list = torch.tensor(Avescore_list).cpu()
        Label_list = torch.tensor(Label_list).cpu()
        Test_lcc = pearsonr(Label_list, Avescore_list)[0]
        Testmae = mean_absolute_error(Label_list, Avescore_list)
        # print(mean_squared_error)
        Testrmse = sqrt(mean_squared_error(Label_list, Avescore_list))
        Test_srcc = spearmanr(Label_list, Avescore_list)[0]


    return accuracy(result_list, groundtruth_list), Test_totalloss / batch_item, \
           Test_lcc, Test_srcc, Testrmse, Testmae,EMD_Total_loss,\
           re_total_loss/batch_item,conf_total / batch_item