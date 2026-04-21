# HW1: 从零开始构建三层神经网络分类器，实现 Fashion-MNIST 图像分类

作业一使用 `NumPy` 从零实现 Fashion-MNIST 分类任务中的 MLP 模型，并手写自动微分与反向传播，不依赖 `PyTorch`、`TensorFlow`、`JAX` 等深度学习框架。命令默认在 `hw1/` 目录下执行。

作业一基于 Fashion-MNIST 原始 IDX 数据实现了数据读取与归一化、训练集/验证集划分、手写自动微分与反向传播的 MLP 模型训练、超参数网格搜索、测试评估与结果可视化的工作流；训练过程中采用交叉熵损失、L2 正则化、SGD 参数更新、学习率衰减与 early stopping 机制，最终输出测试集性能、混淆矩阵、错例分析以及神经元权重可视化结果。

### 1. 环境依赖

依赖 Python `3.9+`、`numpy`、`matplotlib`，安装命令：

```bash
pip install -r requirements.txt
```

### 2. 数据准备

代码从 `data/raw/` 读取 Fashion-MNIST 原始 IDX 文件，目录下应包含以下 4 个文件：

```text
data/raw/train-images-idx3-ubyte
data/raw/train-labels-idx1-ubyte
data/raw/t10k-images-idx3-ubyte
data/raw/t10k-labels-idx1-ubyte
```

- 代码读取解压后的 IDX 文件，而不是 `*.gz` 文件。
- 训练时会将训练集与验证集划分索引保存到 `data/splits/split_indices.npz`。

### 3. 项目结构

- `configs`：训练、测试与超参数搜索所使用的配置文件。
- `data`：Fashion-MNIST 数据及训练/验证划分索引。
- `src`：模型实现、训练脚本、测试脚本与工具函数。
- `artifacts`：训练、测试与搜索过程中生成的权重、日志和图像结果。
- `reports`：实验报告中使用的可视化图片。

主要文件描述：

- `src/tensor.py`：实现 `Tensor`、计算图与反向传播。
- `src/modules.py`：定义 `Linear` 和 MLP 模型。
- `src/losses.py`：交叉熵损失与 L2 正则项。
- `src/dataloader.py`：读取 Fashion-MNIST、划分训练/验证集、生成 mini-batch。
- `src/train.py`：训练主流程，包含 early stopping 和 learning rate decay。
- `src/test.py`：加载最优权重，在测试集上评估并输出混淆矩阵。
- `src/search.py`：基于配置文件做网格搜索。
- `src/grad_check.py`：数值梯度检查。
- `src/report.py`：生成错例图和神经元可视化图。
- `src/utils.py`：配置读取、随机种子、保存模型、绘图等通用函数。

### 4. 训练和测试脚本

梯度检查：

```bash
python -m src.grad_check
```

超参数搜索： 

```bash
python -m src.search
```

输出目录：`artifacts/search/`

* 训练

```bash
python -m src.train
```

输出目录：

- `artifacts/checkpoints/`
- `artifacts/logs/`
- `artifacts/plots/`

测试：

```bash
python -m src.test
```

输出目录：`artifacts/eval/`


### 6. 模型参数保存

- 训练得到的最佳模型权重：`artifacts/checkpoints/best.npz`
- 对应的模型配置信息：`artifacts/logs/best.json`
- GitHub Repo 链接：`https://github.com/hA0oooooo/FDU-computer-vision-26spring/tree/main/hw1`
- 模型权重下载链接：`https://drive.google.com/file/d/1Yh26iWBSMiOriqYB7lu_EKz0cUhys_Z8/view?usp=drive_link`
