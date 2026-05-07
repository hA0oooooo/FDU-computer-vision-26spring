# 任务 1：ResNet-34 Oxford-IIIT Pet 37 类宠物识别

本目录实现期中任务 1：使用 ImageNet 预训练的 ResNet-34 在 Oxford-IIIT Pet Dataset 上完成 37 类宠物分类。实验覆盖四条线：

1. ResNet-34 pretrained baseline
2. learning rate / epoch 小规模超参数分析
3. ResNet-34 scratch ablation
4. ResNet-34 + SE-block attention 对比

代码不加入随机数据增强，不重写 ResNet-34，不在模型末尾加 softmax，不写死 CUDA 版本，不默认多卡。

## 环境安装

本项目不在代码中指定 CUDA 版本，只通过以下逻辑判断设备：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

为了复现实验，建议本地和服务器使用同一组 Python / torch / torchvision 版本。RTX 4060 本地环境建议优先使用 PyTorch 官方稳定 CUDA 12.6 wheel：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

服务器 A6000 即使 `nvidia-smi` 显示 CUDA 13.0，也建议先尝试官方稳定 wheel，例如 `cu126`，或根据 PyTorch 官方 selector 选择稳定支持的 CUDA 版本。不要盲目安装 nightly CUDA 13。`nvidia-smi` 中的 CUDA version 主要表示驱动支持的最高 CUDA runtime 能力，不等同于必须安装完全相同版本的 PyTorch wheel；新版 NVIDIA driver 通常可以运行使用旧版 CUDA Toolkit 构建的应用。

安装后验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`requirements.txt` 不包含 `torch` 和 `torchvision`，避免覆盖已有 CUDA 环境。

参考：

- PyTorch Get Started: https://pytorch.org/get-started/locally/
- NVIDIA CUDA Compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/

## 数据准备

数据集位于仓库顶层：

```text
dataset/oxford-iiit-pet/
├── annotations.tar.gz
└── images.tar.gz
```

配置中的 `data.root` 写为 `../dataset`，因为 `torchvision.datasets.OxfordIIITPet` 会在该父目录下使用 `oxford-iiit-pet/` 子目录。首次运行时 torchvision 会检查并解压官方数据文件。

训练和验证集来自官方 `trainval` split，并按固定 seed 划分；测试集使用官方 `test` split。固定划分文件会保存到：

```text
../dataset/oxford-iiit-pet/splits/
```

统一确定性 transform：

```text
Resize((224, 224)) -> ToTensor -> ImageNet Normalize
```

不使用 `RandomRotation`、`RandomHorizontalFlip`、`RandomResizedCrop`、`ColorJitter` 等随机增强。

## 训练 pretrained baseline

在 `mid1/` 下运行：

```bash
python -m src.train --config configs/resnet34_pretrained.yaml
python -m src.eval --config configs/resnet34_pretrained.yaml
python -m src.plot --history outputs/resnet34_pretrained/history.csv --save outputs/resnet34_pretrained/figures/learning_curve.png
```

设置要点：

- Model: `torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)`
- Output head: `nn.Linear(model.fc.in_features, 37)`
- Loss: `CrossEntropyLoss`
- Optimizer: AdamW
- Backbone LR: `0.0001`
- Head LR: `0.001`
- Epochs: `20`

## 训练 scratch ablation

```bash
python -m src.train --config configs/resnet34_scratch.yaml
python -m src.eval --config configs/resnet34_scratch.yaml
python -m src.plot --history outputs/resnet34_scratch/history.csv --save outputs/resnet34_scratch/figures/learning_curve.png
```

该实验使用 `weights=None`，用于比较 ImageNet 预训练对收敛速度和准确率的影响。

## 训练 SE attention

```bash
python -m src.train --config configs/resnet34_se.yaml
python -m src.eval --config configs/resnet34_se.yaml
python -m src.plot --history outputs/resnet34_se/history.csv --save outputs/resnet34_se/figures/learning_curve.png
```

SE-block 插入在每个 torchvision ResNet BasicBlock 输出之后：

```text
原 BasicBlock: x -> block(x)
SE version:   x -> block(x) -> SE(block(x))
```

实现顺序为：先创建 ImageNet pretrained ResNet-34，再插入 SE-block，最后替换 `fc` 输出层。这样保留原始 ResNet-34 结构和预训练权重，只新增 SE 模块参数。

## 运行超参数搜索

```bash
python -m src.search
```

搜索空间来自 `configs/search.yaml`：

```yaml
backbone_lr: [0.00001, 0.0001]
head_lr: [0.0003, 0.001]
epochs: [10, 20]
```

输出：

```text
outputs/search/search_summary.csv
```

测试集不参与调参，只用于最终评估。

## W&B 可视化

核心训练命令默认不依赖 W&B。若实验报告需要 W&B 截图，可以额外安装：

```bash
pip install wandb
wandb login
```

然后将对应配置文件中的：

```yaml
logging:
  wandb: false
```

改为：

```yaml
logging:
  wandb: true
  entity: 2716504943-fudan-university-school-of-management
  project: mid1-oxford-pet
```

其中 `entity` 是 W&B 账号或团队名，`project` 是本实验在 W&B 中的项目名。训练时会记录 `train/loss`、`val/loss`、`train/accuracy`、`val/accuracy` 和各参数组 learning rate。分类任务主要报告 Accuracy；mAP 是检测任务常用指标，本任务不额外实现检测式 mAP。

注意：`python -m src.plot ...` 生成的是本地 Matplotlib 学习曲线，适合放入报告或 GitHub；报告要求的 W&B 截图需要开启 `logging.wandb: true` 后，从 W&B 页面截取训练/验证 loss 和 accuracy 曲线。

## 输出文件说明

每个主实验会生成：

```text
outputs/resnet34_pretrained/history.csv
outputs/resnet34_pretrained/summary.json
outputs/resnet34_pretrained/checkpoints/best.pt
outputs/resnet34_pretrained/checkpoints/last.pt
outputs/resnet34_pretrained/figures/learning_curve.png
outputs/resnet34_pretrained/eval/test_result.csv

outputs/resnet34_scratch/history.csv
outputs/resnet34_scratch/summary.json
outputs/resnet34_scratch/checkpoints/best.pt
outputs/resnet34_scratch/checkpoints/last.pt
outputs/resnet34_scratch/figures/learning_curve.png
outputs/resnet34_scratch/eval/test_result.csv

outputs/resnet34_se/history.csv
outputs/resnet34_se/summary.json
outputs/resnet34_se/checkpoints/best.pt
outputs/resnet34_se/checkpoints/last.pt
outputs/resnet34_se/figures/learning_curve.png
outputs/resnet34_se/eval/test_result.csv

outputs/search/search_summary.csv
```

`history.csv` 字段：

```text
epoch, train_loss, train_acc, val_loss, val_acc, lr_backbone, lr_head, lr_attention
```

`test_result.csv` 字段：

```text
experiment_name, checkpoint, test_loss, test_acc, total_params, trainable_params
```

`summary.json` 会记录 `best_epoch`、`best_val_acc`、最后一轮指标和 checkpoint 路径；训练结束时终端也会打印 best epoch。

训练权重默认保存到各实验的 `checkpoints/` 子目录，并被 `mid1/.gitignore` 忽略；学习曲线图片默认保存到 `figures/` 子目录，可按课程要求提交到 public GitHub repo 或放入实验报告。`history.csv`、`config.json`、`eval/test_result.csv`、`search_summary.csv` 属于小文件，也可按课程需要提交。

训练好的模型权重不要上传 GitHub，应上传到百度云、Google Drive 等网盘。实验报告中需要写明：

```text
代码 GitHub repo 链接：
模型权重网盘下载地址：
```

## 本地 4060 / 服务器 A6000 建议

- 默认单卡运行。
- 本地 RTX 4060：`batch_size` 可用 16 或 32；若 OOM，手动改为 8。
- 服务器 A6000：`batch_size` 可用 32 或 64。
- `num_workers` 默认 4。
- 如果 Windows 本地 DataLoader 报错，再手动把配置里的 `num_workers` 改为 0；代码中不写 Windows 条件判断。

## 报告建议

报告主结果表建议包含：

```text
Model | Pretrained | Attention | Backbone LR | Head LR | Epochs | Best Val Acc | Test Acc | Params
```

讨论重点：

1. ImageNet pretrained ResNet-34 baseline 的 Val/Test Accuracy
2. Scratch ResNet-34 是否收敛更慢、准确率更低
3. `backbone_lr`、`head_lr`、`epochs` 对性能的影响
4. SE-block 是否带来提升
5. 新输出头使用较大学习率、backbone 使用较小学习率的原因
