说明：
Transformer-based Cross-modal Multi-contrast Network for Ophthalmic Diseases Diagnosis的代码
在OCTA-300数据集上运行
根目录保存所有运行的程序文件
lib文件夹中保存所需函数文件

下载数据集：
https://ieee-dataport.org/open-access/octa-500
解压文件
创建文件夹命名为OCTA-300
将octa_3m_oct_part1至octa_3m_oct_part4合并到OCTA-300\OCT文件夹
将octa-500_ground_truth\OCTA-500\OCTA_3M\Projection Maps\OCT(FULL)文件夹复制到OCTA-300\中
将octa-500_ground_truth\OCTA-500\OCTA_3M\Text labels.xlsx复制到OCTA-300\中

运行程序：
1 运行dataprocess.py
修改data_path为建立的OCTA-300\文件夹路径
将saveroot指定为保存的压缩文件路径
生成两个文件：
data3m.npz（所有数据）
data3m_train.npz（5倍交叉验证的训练数据）

2 运行CFP_supcon.py:
在训练集上进行CFP有监督对比学习

3 运行Bscan_contrast.py:
在训练集上进行Bscan动量监督对比学习

4 运行OCT_supcon.py:
在训练集上进行OCT有监督对比学习

5 运行Multi_supcon.py:
在训练集上进行跨模态对比学习

6 运行Head_train.py:
在训练集上训练分类头

7 运行Head_test.py:
测试分类性能
