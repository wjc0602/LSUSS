# Jittor 大规模无监督语义分割

## 简介

![image](https://user-images.githubusercontent.com/20515144/196449430-5ac6a88c-24ea-4a82-8a45-cd244aeb0b3b.png)

PASS是一种新的大规模无监督语义分割方法，包含四个步骤。1） 通过基于代理任务的自监督表征学习方法（即非对比的像素级表征对齐策略，和自深到浅的监督策略）训练随机初始化的模型，以学习形状和类别表征。在表征学习之后，我们获得所有训练图像的特征集。2） 通过基于像素注意力的聚类方法获得伪类别，并将生成的类别分配给每个图像像素。3） 用生成的伪标签对预训练的模型进行微调，以提高分割质量。4） 在推理过程中，LUSS模型将生成的标签分配给图像的每个像素，与有监督模型相同。

#### 运行环境

* 系统: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, 或者 **Windows Subsystem of Linux (WSL)**
* Python版本 >= 3.7
* CPU 编译器
    * g++ (>=5.4.0)
    * clang (>=8.0)


#### 安装依赖

## 第一步: 安装计图
计图的安装可以参考以下文档[Jittor install](https://github.com/Jittor/jittor#install)

## 第二步: 安装依赖
```shell
python -m pip install scikit-learn
python -m pip install pandas
python -m pip install munkres
python -m pip install tqdm
python -m pip install pillow
python -m pip install opencv-python
python -m pip install faiss-gpu
```

## 训练

单卡训练可运行以下命令：
```
sh train.sh
```
## 推理

生成测试集上的结果可以运行以下命令：

```
sh test.sh
```

## 致谢

此项目基于论文 *Large-scale Unsupervised Semantic Segmentation* 实现