import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as trans
import warnings
warnings.filterwarnings('ignore')
from lib.ViTme import build_mlp, fc
import lib.Head_loader2
import lib.readData
from sklearn.metrics import confusion_matrix
import itertools
from lib.ViTme import ViT, ViT3D
import lib.Bscan_loader2
import lib.Multi_loader2
import argparse

def parse_option():
    parser = argparse.ArgumentParser(description='Argument for testing')
    parser.add_argument('--saveroot', type=str, default='../data3m.npz',
                        help='Packed files after processing')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_option()

    class_name = ['NORMAL', 'CNV', 'DR', 'AMD', 'CSC', 'RVO', 'OTHERS']  # 前四个是3M中使用的，

    # 读取所有数据
    data2d, datasetlist, labs, clist, Bclist = lib.readData.read_datasetme(opt.saveroot)  # 读取文件路径和诊断真值
    augmentation = trans.Compose([
        trans.ToTensor(),
    ])

    #一提取Bscan特征用Bscan_contrast中保存的模型
    if os.path.isfile(os.path.join('save/Bscan', "datalogits_all.npz")):
        f = np.load(os.path.join('save/Bscan', "datalogits_all.npz"), allow_pickle=True)
        Bscanlogits = f['arr_0'][()] # ndarray转化为内置字典类型dict
    else:
        pretrained = 'save/Bscan/model_best.pth.tar'
        model = ViT(
            image_size = 304,
            patch_size = 38,
            dim=768,
            channels=1,
            depth = 6,
            heads = 6,
            mlp_dim = 1024,
            outdim= 512,)
        del model.head
        model.head = model._build_mlp(3, 512, 1024, 512)

        # load from pre-trained, before DistributedDataParallel constructor
        if pretrained:
            if os.path.isfile(pretrained):
                print("=> loading checkpoint '{}'".format(pretrained))
                checkpoint = torch.load(pretrained, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # print(k)
                    # retain only base_encoder up to before the embedding layer 只保留base_encoder，并且去掉head
                    if k.startswith('module.momentum_encoder'):
                        # remove prefix
                        state_dict[k[len("module.momentum_encoder."):]] = state_dict[k]  # 只保留base_encoder.之后的有效字
                    # delete renamed or unused k
                    del state_dict[k]  # 删除列表中的元素

                msg = model.load_state_dict(state_dict, strict=False)
                print("=> loaded pre-trained model '{}'".format(pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(pretrained))

        # model = baidu_lib.Model()
        device = torch.device("cuda:0")
        model = torch.nn.DataParallel(model)
        model.to(device)

        augmentation = trans.Compose([
            trans.ToTensor(),
        ])

        test_loader = lib.Bscan_loader2.DataLoader(
            dataset=[datasetlist, labs, clist, Bclist],
            img1_trans=augmentation,
            img2_trans=augmentation,
            batch_size=20,
            num_workers=0,
            splfun='suqs',
        )

        all_lab = []
        all_log = []
        Bscanlogits = {}
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                # load batch
                names = data[0]
                oct_imgs = data[1].to(device=device, dtype=torch.float32)
                labels = data[2].cpu().numpy()
                logits = model(oct_imgs).cpu().numpy()
                for ib in range(20):
                    Bscanlogits.update({names[ib]: logits[ib, :]})  # update增加子列表
        np.savez(os.path.join('save/Bscan', "datalogits_all.npz"), Bscanlogits)

    #二用Multi_supcon中保存的模型提取OCT和CFP特征
    model1 = ViT3D(
        num_patches=304,
        patch_dim=512,
        num_classes=4,
        dim=768,
        depth=2,
        channels=1,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    del model1.head
    model1.head = model1._build_mlp(3, 768, 1024, 512)

    model2 = ViT(
        image_size=304,
        patch_size=38,
        dim=768,
        channels=1,
        depth=6,
        heads=6,
        mlp_dim=1024,
        outdim=512)

    # model = baidu_lib.Model()
    device = torch.device("cuda:0")
    model1 = torch.nn.DataParallel(model1)
    model1.to(device)
    model2 = torch.nn.DataParallel(model2)
    model2.to(device)

    pretrained = 'save/Multi/' + 'best_ViT3D.pth'
    # load from pre-trained, before DistributedDataParallel constructor
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['model']
        msg = model1.load_state_dict(state_dict, strict=False)
        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))

    pretrained = 'save/Multi/' + 'best_ViT.pth'
    # load from pre-trained, before DistributedDataParallel constructor
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['model']
        msg = model2.load_state_dict(state_dict, strict=False)
        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))

    test_loader = lib.Multi_loader2.DataLoader2(
        dataset=[datasetlist, labs, clist, Bscanlogits],
        img1_trans=augmentation,
        img2_trans=augmentation,
        batch_size=40,
        num_workers=0,
        splfun='Sequen'
    )

    octlog = []
    cfplog = []
    labs = []

    model1.eval()
    model2.eval()

    device = torch.device("cuda:0")
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # load batch
            # fundus_imgs = data[0].to(device=device, dtype=torch.float32)
            oct_imgs = data[0].to(device=device, dtype=torch.float32)
            cfp_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].cpu().numpy()
            logits1 = model1(oct_imgs).cpu().numpy()
            logits2 = model2(cfp_imgs).cpu().numpy()
            for ib in range(len(labels)):
                octlog.append(logits1[ib])
                cfplog.append(logits2[ib])
                labs.append(labels[ib])

    octlog = np.stack(octlog)
    cfplog = np.stack(cfplog)
    labs = np.stack(labs)

    np.savez(os.path.join('save/Multi', "allmodallogits_all.npz"), octlog, cfplog, labs)

    #三将生成的oct特征cfp特征传输给训练好的CFH
    img_transforms = trans.Compose([
        trans.ToTensor(),
    ])

    # 设置训练数据加载器
    train_loader = lib.Head_loader2.DataLoader(
        dataset=[octlog, cfplog, clist, labs],
        img1_trans=img_transforms,
        img2_trans=img_transforms,
        batch_size=20,
        num_workers=0
    )

    mlp1 = build_mlp(3, 512, 1024, 512)
    mlp2 = build_mlp(3, 512, 1024, 512)
    clh = fc(512,4)

    files= os.listdir('save/Head/') #得到文件夹下的所有文件名称
    for mdname in files: #遍历文件夹
        if mdname.endswith('.pth'):  # 如果检测到模型
            mdid = mdname.split('_')[2][:-4]
            path_experiment='save/Head/'+mdid+'/'

            if not os.path.isdir(path_experiment):
                os.makedirs(path_experiment)

            #加载模型
            state_dict = torch.load('save/Head/'+mdname, map_location=torch.device('cpu'))
            msg = mlp1.load_state_dict(state_dict['mlp1'])
            msg = mlp2.load_state_dict(state_dict['mlp2'])
            msg = clh.load_state_dict(state_dict['clh'])

            # model = baidu_lib.Model()
            device = torch.device("cuda:0")
            mlp1.to(device)
            mlp2.to(device)
            clh.to(device)

            #测试
            mlp1.eval()
            mlp2.eval()
            clh.eval()

            sm = torch.nn.Softmax(dim=1)    #沿通道方向

            gt = []
            predict = []
            #learning_rates = AverageMeter()

            with torch.no_grad():
                for i, data in enumerate(train_loader, 0):
                    #读数据
                    oct_imgs = data[0].to(device=device, dtype=torch.float32)
                    cfp_imgs = data[1].to(device=device, dtype=torch.float32)
                    labels = data[2].to(device=device, dtype=torch.long)
                    bsz = labels.shape[0]
                    #前向传播
                    oct_log = mlp1(oct_imgs)
                    cfp_log = mlp2(cfp_imgs)
                    log = torch.stack([oct_log,cfp_log],dim=1)
                    log = sm(log)
                    oct_ccon = log[:, 0, :]
                    cfp_ccon = log[:, 1, :]
                    oct_con = oct_ccon.mean(axis=1)
                    cfp_con = cfp_ccon.mean(axis=1)
                     #log = (oct_log+cfp_log)/2
                    flog = torch.mul(oct_ccon, oct_log)+torch.mul(cfp_ccon, cfp_log)
                    pre = clh(flog)
                    predicted = torch.max(pre.data, dim=1).indices
                    for idx, _ in enumerate(labels):
                        gt.append(labels[idx])
                        predict.append(predicted[idx])

                ground_truth = np.array([g.item() for g in gt])
                prediction = np.array([pred.item() for pred in predict])

                title='Confusion matrix'
                cm = confusion_matrix(ground_truth, prediction)
                cmap=plt.cm.Blues
                print(cm)
                plt.figure()
                plt.imshow(cm, interpolation='nearest', cmap=cmap)
                plt.title(title)
                plt.colorbar()
                tick_marks = np.arange(len(class_name))
                plt.xticks(tick_marks, class_name, rotation=45)
                plt.yticks(tick_marks, class_name)
                fmt = 'd'
                thresh = cm.max() / 2.
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, i, format(cm[i, j], fmt),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.tight_layout()
                plt.savefig(path_experiment + "confusion matrix.jpg", bbox_inches='tight')
                #plt.show()

                file_perf = open(path_experiment + 'performances.txt', 'w')
                # Sensitivity, Specificity and F1 per class
                print('class acc, sen, spe, pre, miou, f1')
                file_perf.write('class acc, sen, spe, pre, miou, f1' + '\n')
                n_classes = 4
                allacc = []
                allsen = []
                allspe = []
                allpre = []
                allmiou = []
                allf1 = []
                for i in range(n_classes):
                    y_test = [int(x == i) for x in ground_truth]  # obtain binary label per class
                    tn, fp, fn, tp = confusion_matrix(y_test, [int(x == i) for x in prediction]).ravel()
                    acc = float(tp + tn) / (tn + fp + fn + tp)
                    sen = float(tp) / (fn + tp)
                    spe = float(tn) / (tn + fp)
                    pre = float(tp) / (tp + fp)
                    miou = float(tp) / (tp + fp + fn)
                    f1 = 2 * pre * sen / (pre + sen)
                    print(class_name[i], '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (acc, sen, spe, pre, miou, f1))
                    file_perf.write(class_name[i]+ '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (acc, sen, spe, pre, miou, f1) + '\n')
                    allacc.append(acc)
                    allsen.append(sen)
                    allspe.append(spe)
                    allpre.append(pre)
                    allmiou.append(miou)
                    allf1.append(f1)
                aacc = sum(allacc) / n_classes
                asen = sum(allsen) / n_classes
                aspe = sum(allspe) / n_classes
                apre = sum(allpre) / n_classes
                amiou = sum(allmiou) / n_classes
                af1 = sum(allf1) / n_classes
                print('mean_of_all', '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (aacc, asen, aspe, apre, amiou, af1))
                file_perf.write('mean_of_all' + '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (aacc, asen, aspe, apre, amiou, af1) + '\n')

                file_perf.close()
                print("done")

if __name__ == '__main__':
    main()