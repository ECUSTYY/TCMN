复制于TCMN1.2：
在OCT300数据集上整理
并且精简程序

修改：
修正在训练多模态分类时，没有用到多模态特征
ViT3D增加SlEm和SlAtt
修改两个模型的head
增加了CFP_linear.py和CFP_test.py
用于单模态线性分类

修改：
下载数据集：
https://ieee-dataport.org/open-access/octa-500
解压文件
创建文件夹命名为OCTA-300
将octa_3m_oct_part1至octa_3m_oct_part4合并到OCTA-300\OCT文件夹
将octa-500_ground_truth\OCTA-500\OCTA_3M\Projection Maps\OCT(FULL)文件夹复制到OCTA-300\中
将octa-500_ground_truth\OCTA-500\OCTA_3M\Text labels.xlsx复制到OCTA-300\中

先运行dataprocess.py
修改data_path为建立的OCTA-300\文件夹路径
将saveroot指定为保存的压缩文件路径
生成两个文件：
data3m.npz（所有数据）
data3m_train.npz（5倍交叉验证的训练数据）

CFP_supcon.py:
CFP在训练集上进行有监督对比学习
梳理使用的opt参数：
opt.saveroot='../data3m_train.npz'
opt.image_size=304
opt.batchsize=20 40
opt.temp=0.07

opt.learning_rate=0.05
opt.momentum=0.9
opt.weight_decay=0.0001

opt.tb_folder='./save/CFP'
opt.epochs=100

opt.save_folder='./save/CFP'
梳理使用的函数：
util
losses
readData
CFPloader
ViTme

Bscan_contrast.py:
Bscan在训练集上动量监督对比学习
梳理使用的opt参数：
opt.saveroot = '../data3m_train.npz'
opt.image_size = 304
opt.batchsize = 320
opt.lr = 0.006
opt.epochs=10
opt.moco_m=0.99
args.gpu=0
梳理使用的函数：
readData
CFPloader
ViTme
Bscan_loader
Bscan_loader2

OCT_supcon.py:
梳理使用的函数：
util
losses
readData
ViTme
OCT_loader
梳理使用的opt参数：
opt.saveroot = '../data3m_train.npz'
opt.batchsize=20 40
opt.temp
opt.learning_rate
opt.momentum
opt.weight_decay
opt.tb_folder
opt.epochs

Multi_supcon.py:
梳理使用的函数：
util
mlosses
readData
ViTme
Multi_loader
Multi_loader2
梳理使用的opt参数：
opt.saveroot = '../data3m_train.npz'
opt.batchsize=40
opt.temp=0.07
opt.learning_rate=0.05
opt.momentum=0.9
opt.weight_decay=1e-4
opt.epochs=50
opt.tb_folder
opt.save_folder

Head_train.py:
梳理使用的函数：
util
readData
ViTme
Head_loader
梳理使用的opt参数：
opt.saveroot = '../data3m_train.npz'
opt.save_folder
opt.batchsize
opt.init_lr = 1e-4
opt.n_epochs

Head_test.py:
梳理使用的函数：
util
readData
ViTme
Head_loader2
Bscan_loader2
Multi_loader2
梳理使用的opt参数：
opt.saveroot = '../data3m_train.npz'

目标：
所有步骤公用一个readData函数
每个步骤有：
一个训练程序
一个测试/嵌入程序
一个dataloader(包括顺序、随机、类平衡)

保存的数据为data3m.npz
修改readme.py
18行：    if not os.path.exists(os.path.join(saveroot,"data3m.npz")):
63行：        np.savez(os.path.join(saveroot, "data3m.npz"), data2d, datasetlist, dedi, clist, Bclist)
66行：        f = np.load(os.path.join(saveroot, "data3m.npz"),allow_pickle=True)
78行：    workBook = xlrd.open_workbook(r'D:\task8\logs\Text labels.xlsx')
95行：    data2d = np.zeros((1, 304, 304, len(datasetlist['FULL'])), dtype=np.int)

程序说明：
整理所有的步骤，分成单独的程序
第一步：CFP的有监督学习，嵌入图
第二步：B-scan自监督学习，嵌入图
第三步：OCT有监督学习，嵌入图
第四步：多模态数据有监督学习，嵌入图
第五步：分类头学习，测试分类结果

第一步：CFP使用有监督对比学习生成的特征分布
CFP_supcon.py:
使用readData、CFPloader、ViTme、losses、util
修改参数：
    image_size = 304
    batchsize=32
    data_path = 'E:\\task8\\OCT500\\OCTA-300\\'
    model = ViT(
    image_size=304,
    patch_size=38,
输出：在save/CFP中
输出模型best.pth和tensorboard记录文件

CFPloader：使用类平衡
修改159行：        for i in range(int(self.batch_size/4*200)):
输出增强图像data[0]、原图data[1]、标签data[2]
有三种模式：类平衡(clbl),顺序(seq),随机(rand)

CFP_tsne.py:
使用readData、CFPloader2、ViTme
修改使用CFPloader
修改    model = ViT(
    image_size=304,
    patch_size=38,
同时计算模型参数量

修改ViTme中的模型：
实现MIA Head功能
        self.InEm = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.InAtt = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Softmax(dim=1)
        )
在MIA Head中实现维度缩减

采用ViT-Base的超参

在模型的输出端使用nn.BatchNorm1d(outdim, affine=False)
否则不收敛
训练时batchsize设的小一点效果更好（试验过64和128效果都不行）

增加超参数选取实验(中间特征维度)：
参考ViT-GAMMA1.1
在每个epoch的训练后，对全局计算监督损失
复制baidu_test.py->val到CFP_supcon2.py
在losses中增加SupConLossall
在每个epcoh的训练后使用val和SupConLossall对全局的嵌入计算损失
使用评估损失保存模型
精简save_model函数，仅保存模型参数
tensorboard --logdir E:\task8\me\ViT-OCT9.2\save\CFP

第二步：Bscan使用自监督对比学习生成的特征分布
Bscan_contrast使用Bscan_loader，CFP_tsne使用CFPloader2
保存在save/Bscan中
训练过程数据，best模型，嵌入图
参考：ViT-OCT4.3
复制baidu_contrast.py到Bscan_contrast.py
复制mydataloader2 .py到Bscan_loader.py
增大batchsize至80，复制下列设置
--lr=1.5e-4 --epochs=300 --warmup-epochs=40 --moco-m-cos --rank=0 --gpu=0
复制baidu_tsne .py到Bscan_tsne .py
修改137行
for i in range(len(self._dataset[3])):
自动根据类别数调整迭代次数
将baidu_Bsqr.py也集成到这个函数中
复制Bscan_loader.py到Bscan_loader2.py，加载的数据：
Bscan文件名，Bscan图像，类别
保存Bscan特征字典datalogits.npz

第三步：CFP使用有监督对比学习生成的特征分布
参考：ViT-OCT6.3
复制main_supcon_thick.py为OCT_supcon.py
修改145行：
 ViT3D( image_size=400,
复制mydataloader.py为OCT_loader.py

如果以tensor形式保存数据则数据量会很大
用以下形式转换为np数组
datalogits2 = {}
for x in range(300):
    for y in range(400):
        datalogits2.update({datasetlist[()]['OCT'][x][y]: datalogits[()][datasetlist[()]['OCT'][x][y]].cpu().numpy()})
np.savez(os.path.join('save/Bscan', "datalogits.npz"), datalogits2)

ViT3D中不用image_size和patch_size
改为  num_patches = 304
        patch_dim = 256

修改batch_size=28

监督对比学习使用
adjust_learning_rate_sup
调整学习率

需要修改OCT_loader中BatchSampler
把4改为7
并且BatchSampler并没有用到RandomSampler


模型保存在save/OCT中

复制baidu_tsne.py到OCT_tsne.py
复制mydataloader2.py为OCT_loader2.py
嵌入图保存在save/OCT/inst_contrast.png

第四步：多模态数据有监督学习
参考ViT-OCT7.3
复制main_supcon_thick.py为Multi_supcon.py
复制mydataloader.py为Multi_loader.py
复制losses.py到mlosses.py
复制baidu_tsne.py到Multi_tsne.py
第140行修改增加的类别数
all_lab.append(labels[ib]+7)
复制mydataloader2.py为Multi_loader2.py

使用batchsize=28
对大类的配对不是你很好
修改305行：
batchsize=42
调高batchsize则可以实现对大类较好的配对
同时发现并不是迭代次数越多越好
在20次左右有着较好的聚类效果
之后大类之间却发散

嵌入图保存在save/Multi/inst_contrast.png

参考baidu_Mtqr.py整合进Multi_tsne.py
将多模态特征保存下来

第五步：分类头学习
参考ViT-OCT8.3
复制head_train.py为Head_train.py
复制mydataloader.py到Head_loader.py
修改66行：clh = fc(256,7)
否则会报错：CUDA error: device-side assert triggered

复制head_test.py为Head_test.py
复制mydataloader5.py到Head_loader2.py
自动测试所有保存的模型

测试：
运行Head_test.py
加载所有数据，
读取保存的CFP、Bscan、OCT模型
对数据生成最终的多模态特征并诊断

结果：
虽然保存了分类头每个epoch的权重，
但除了1.5850有一个误分类
其他都是过拟合