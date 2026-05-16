# 任务 1: ResNet-34 Oxford-IIIT Pet 37 类宠物识别

本任务在 Oxford-IIIT Pet Dataset 上实现 37 类宠物分类。代码默认在 `mid1/` 目录下执行，使用 PyTorch / torchvision / timm 构建训练、验证、测试、超参数搜索和可视化工作流。

本目录实现了五组模型实验：ImageNet 预训练 ResNet-34 baseline、ResNet-34 scratch 消融、ResNet-34 + SE-block attention、ViT-Tiny pretrained、Swin-T pretrained。其中 ResNet-34 baseline、SE、ViT-Tiny 和 Swin-T 都是在 ImageNet 预训练权重基础上替换 37 类分类头后进行微调；ResNet-34 scratch 不加载 ImageNet 权重，用于比较预训练对收敛和泛化的影响。训练脚本会记录训练集 loss、训练集 accuracy、验证集 loss、验证集 accuracy，并通过 W&B 生成训练过程曲线，供实验报告截图使用。测试脚本会加载各实验的最佳 checkpoint，并将测试集结果统一追加。

- GitHub Repo：`https://github.com/hA0oooooo/FDU-computer-vision-26spring/tree/main/mid1`

- 模型权重下载：`https://drive.google.com/drive/folders/1gmX55mTGyOlxCYiSGrda_7rzbWHlcGmd?usp=sharing`


### 1. 环境依赖

建议使用 Python `3.9+`，PyTorch 与 CUDA 版本单独安装，避免 `requirements.txt` 覆盖已有 CUDA 环境。本地 RTX 4060 优先使用 PyTorch 官方 CUDA 12.6 wheel：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

训练过程使用 W&B 记录曲线，W&B 项目和 entity 在 `configs/*.yaml` 中配置。首次使用前需登录：

```bash
wandb login
```


### 2. 数据准备

数据集位于仓库顶层 `dataset/oxford-iiit-pet/`：

```text
dataset/oxford-iiit-pet/
├── annotations.tar.gz
└── images.tar.gz
```

数据划分方式：

- 使用官方 `trainval` split，再按固定 seed 划分训练集和验证集。
- 使用官方 `test` split 作为最终测试集。
- 固定划分索引保存到 `../dataset/oxford-iiit-pet/splits/trainval_seed42_val0.2.json`。


### 3. 项目结构

- `configs`：各模型训练配置和超参数搜索配置
- `src`：数据集、模型、训练、测试、搜索和绘图代码
- `outputs`：训练权重、训练日志、搜索结果和测试汇总结果
- `reports`：实验报告中使用的图像或中间材料

主要文件说明：

- `src/dataset.py`：固定 train/val 划分和 DataLoader
- `src/models.py`：构建 ResNet-34、ViT-Tiny、Swin-T，并替换 37 类分类头
- `src/attention.py`：实现 SE-block，并将其插入 ResNet-34 BasicBlock 之后
- `src/train.py`：训练主流程，并记录 W&B 曲线
- `src/eval.py`：加载最佳 checkpoint 进行评估，并追加写入 `outputs/eval.csv`
- `src/search.py`：基于 `configs/search.yaml` 做超参数组合搜索
- `src/wandb_utils.py`：W&B 初始化与日志记录
- `src/utils.py`：工具函数

五个模型配置文件：

- `configs/resnet34_pretrained.yaml`：基于 ImageNet 预训练的 ResNet-34 微调作为基线
- `configs/resnet34_scratch.yaml`：随机初始化的 ResNet-34 训练作为消融实验
- `configs/resnet34_se.yaml`：基于 ImageNet 预训练的 ResNet-34 + SE-block attention 微调
- `configs/vit_tiny_pretrained.yaml`：基于 ImageNet 预训练的 ViT-Tiny 微调
- `configs/swin_t_pretrained.yaml`：基于 ImageNet 预训练的 Swin-T 微调

每个实验的具体训练参数以对应 `configs/*.yaml` 为准。

### 4. 训练与测试

以下命令均在 `mid1/` 目录下执行。训练五个模型：

```bash
python -m src.train --config configs/resnet34_pretrained.yaml
python -m src.train --config configs/resnet34_scratch.yaml
python -m src.train --config configs/resnet34_se.yaml
python -m src.train --config configs/vit_tiny_pretrained.yaml
python -m src.train --config configs/swin_t_pretrained.yaml
```

训练输出位于各自的 `output_dir` 下。测试五个模型：

```bash
python -m src.eval --config configs/resnet34_pretrained.yaml
python -m src.eval --config configs/resnet34_scratch.yaml
python -m src.eval --config configs/resnet34_se.yaml
python -m src.eval --config configs/vit_tiny_pretrained.yaml
python -m src.eval --config configs/swin_t_pretrained.yaml
```

超参数搜索：

```bash
python -m src.search
```

搜索配置位于 `configs/search.yaml`，当前基于 `configs/resnet34_pretrained.yaml` 做 pretrained ResNet-34 微调搜索，比较不同 `backbone_lr`、`head_lr`、`batch_size` 的组合。搜索结果保存到：

```text
outputs/search/search_summary.csv
```
