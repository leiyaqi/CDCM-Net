

import torch
import numpy as np
from DataSet import MyData
from torch.utils.data import DataLoader
from Encoder import MMencoder
from TransformerEncoder import MFBFusion,CCFusion,MMDF,ImgEncoder
from MemoryNetWork import Memory

from Train import MyTrain
from Test import MyTest

import visdom


ismemroy = True



def dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for id, text, label in batch:
        images.append(id)
        bboxes.append(text)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, bboxes, labels


if __name__ == '__main__':
    train_name_index = []
    train_text_list = []
    train_label_list = []
    test_text_list = []
    test_name_index = []
    test_label_list = []
    val_text_list = []
    val_name_index = []
    val_label_list = []
    test_text = torch.load("3AVA_test_text_dict.pth")
    test_label = torch.load("3AVA_test_label_dict.pth")
    train_text = torch.load("3AVA_train_text_dict.pth")
    train_label = torch.load("3AVA_train_label_dict.pth")


    for item in test_text:
        test_name_index.append(item)
        test_text_list.append(test_text[item])
        test_label_list.append(test_label[item])

        train_name_index.append(item)
        train_text_list.append(test_text[item])
        train_label_list.append(test_label[item])


    # for item in train_text:
    #
    #     train_name_index.append(item)
    #     train_text_list.append(train_text[item])
    #     train_label_list.append(train_label[item])

    test_data = list(zip(test_name_index, test_text_list, test_label_list))
    # train_data = list(zip(train_name_index, train_text_list, train_label_list))
    train_data = list(zip(train_name_index, train_text_list, train_label_list))
    test_data = np.array(test_data, dtype=object)
    train_data = np.array(train_data, dtype=object)


    print("load train sample", len(train_data), "test samples:", len(test_data))
    #  dataset dataloder
    train_set = MyData(train_data)
    test_set = MyData(test_data)

    train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True, collate_fn=dataset_collate,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, collate_fn=dataset_collate,
                             drop_last=True)


    epochs_num = 1
    best_acc = 0.0

    mmencoder = MMencoder().cuda()
    mmfusion = MMDF().cuda()
    # mmfusion = ImgEncoder().cuda()
    # mmfusion = MMFusion().cuda()
    # mmfusion =CCFusion().cuda()
    # mmfusion = MFBFusion().cuda()
    mmfusion.load_state_dict(torch.load("0.87AVA_mem_conf_mmfusion.pth"))
    memory = None
    viz = visdom.Visdom()
    if ismemroy:
        memory = Memory().cuda()
        memory.load_state_dict(torch.load("0.87AVA_mem_conf_memory.pth"))
        viz.line([0.], [0], win='2Vacc', opts=dict(title='2Vacc'))
        viz.line([0.], [0], win='2Vlcc', opts=dict(title='2Vlcc'))
        viz.line([0.], [0], win='2Vsrcc', opts=dict(title='2Vsrcc'))
        viz.line([0.], [0], win='2Vloss', opts=dict(title='2Vloss'))
        # viz.line([0.], [0], win='Vtrainrmse', opts=dict(title='Vtrainrmse'))
        # viz.line([0.], [0], win='Vtrainmae', opts=dict(title='Vtrainmae'))
        viz.line([0.], [0], win='2Vrmse', opts=dict(title='2Vrmse'))
        viz.line([0.], [0], win='2Vmae', opts=dict(title='2Vmae'))
        # viz.line([0.], [0], win='Vtrainloss', opts=dict(title='Vtrainloss'))
        # viz.line([0.], [0], win='Vtrainlcc', opts=dict(title='Vtrainlcc'))
        # viz.line([0.], [0], win='Vtrainsrcc', opts=dict(title='Vtrainsrcc'))
        # viz.line([0.], [0], win='VtrainAcc', opts=dict(title='VtrainAcc'))
    else:
        viz.line([0.], [0], win='Racc', opts=dict(title='Racc'))
        viz.line([0.], [0], win='Rlcc', opts=dict(title='Rlcc'))
        viz.line([0.], [0], win='Rsrcc', opts=dict(title='Rsrcc'))
        viz.line([0.], [0], win='Rloss', opts=dict(title='Rloss'))
        # viz.line([0.], [0], win='Rtrainrmse', opts=dict(title='Rtrainrmse'))
        # viz.line([0.], [0], win='Rtrainmae', opts=dict(title='Vtrainmae'))
        viz.line([0.], [0], win='Rrmse', opts=dict(title='Rrmse'))
        viz.line([0.], [0], win='Rmae', opts=dict(title='Rmae'))
        # viz.line([0.], [0], win='Rtrainloss', opts=dict(title='Vtrainloss'))
        # viz.line([0.], [0], win='Rtrainlcc', opts=dict(title='Vtrainlcc'))
        # viz.line([0.], [0], win='Rtrainsrcc', opts=dict(title='Vtrainsrcc'))
        # viz.line([0.], [0], win='RtrainAcc', opts=dict(title='RtrainAcc'))

    #save best epoch as mybestmodule
    for epoch in range(epochs_num):
        print("epoch:", epoch)

        # trainloss = MyTrain(epoch, mmencoder,memory, mmfusion,train_loader,
        #                                           ismemroy) 
        # print("train loss :",trainloss)

        TestAcc, Testloss, Testlcc, Testsrcc, Testrmse, Testmae,\
            EMD_Total_loss,re_total_loss,conf_loss = MyTest(epoch, mmencoder,memory, mmfusion,
                                                                         test_loader, ismemroy)
        print('TestAcc', TestAcc, "emdloss",EMD_Total_loss, "reloss", re_total_loss, "conf_loss", conf_loss)
        # print("lcc", Testlcc, "srcc", Testsrcc, "rmse", Testrmse, "mae", Testmae)
        if TestAcc > best_acc:
            if ismemroy:
                torch.save(memory.state_dict(), str(round(TestAcc, 2)) + 'AVA_mem_conf_memory.pth')
                torch.save(mmfusion.state_dict(), str(round(TestAcc, 2)) + 'AVA_mem_conf_mmfusion.pth')

            best_acc = TestAcc
        if ismemroy:
            viz.line([TestAcc], [epoch + 1], win='2Vacc', update='append',
                     opts=dict(title='2Vacc'))
            viz.line([Testlcc], [epoch + 1], win='2Vlcc', update='append',
                     opts=dict(title='2Vlcc'))
            viz.line([Testsrcc], [epoch + 1], win='2Vsrcc', update='append',
                     opts=dict(title='2Vsrcc'))
            viz.line([Testloss], [epoch + 1], win='2Vloss', update='append',
                     opts=dict(title='2Vloss'))
            viz.line([Testrmse], [epoch + 1], win='2Vrmse', update='append',
                     opts=dict(title='2Vrmse'))
            viz.line([Testmae], [epoch + 1], win='2Vmae', update='append',
                     opts=dict(title='2Vmae'))
        else:
            viz.line([TestAcc], [epoch + 1], win='Racc', update='append',
                     opts=dict(title='Racc'))
            viz.line([Testlcc], [epoch + 1], win='Rlcc', update='append',
                     opts=dict(title='Rlcc'))
            viz.line([Testsrcc], [epoch + 1], win='Rsrcc', update='append',
                     opts=dict(title='Rsrcc'))
            viz.line([Testloss], [epoch + 1], win='Rloss', update='append',
                     opts=dict(title='Rloss'))
            viz.line([Testrmse], [epoch + 1], win='Rrmse', update='append',
                     opts=dict(title='Rrmse'))
            viz.line([Testmae], [epoch + 1], win='Rmae', update='append',
                     opts=dict(title='Rmae'))

