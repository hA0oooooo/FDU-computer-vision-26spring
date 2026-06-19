# 基于 2DGS 与 AIGC 的多源资产生成与真实场景融合

## 摘要

本实验完成了一个从资产生成到场景融合的完整 3D 视觉流程。我们使用三条不同技术路线构建独立物体资产：Object A 使用手机多视角照片，通过 COLMAP 估计相机位姿并使用 2D Gaussian Splatting（2DGS）重建；Object B 使用 threestudio / DreamFusion，仅依赖文本 prompt 和 SDS loss 生成蘑菇形虚拟物体；Object C 使用 Magic123，从单张兰花照片生成带纹理的 3D mesh。同时，我们使用 Mip-NeRF 360 的 `garden` 场景作为真实背景，并用 2DGS 重建。最终将 Object A/B/C 与 Garden 背景统一转换为显式 mesh 或 mesh 代理，在 Blender 中进行尺度、位置、材质和相机轨迹融合，输出多视角漫游视频。

实验结果表明，多视角重建在几何和纹理一致性上最可靠，但依赖足够密集的输入视角；文本到 3D 可以快速生成语义明确的虚拟资产，但几何细节和局部拓扑更不稳定；单图到 3D 在保留输入视图外观方面较好，但背面和细结构依赖扩散先验，容易出现想象性补全。整体流程证明，将不同 3D 表达最终统一到 Blender 可处理的 mesh 表达，是本任务中最稳健、工程成本最低的融合方案。

- GitHub Repo：`https://github.com/hA0oooooo/FDU-computer-vision-26spring/tree/main/final1`
- 模型权重与数字资产下载：`TODO`


## 1. 任务与数据

本任务要求完成三个物体资产和一个背景场景的构建，并将它们融合到同一个 3D 场景中。表 1 总结了四类资产的来源、输入和输出形式。

| 资产 | 技术路线 | 输入 | 主要输出 | 用途 |
|---|---|---|---|---|
| Garden | 2DGS 背景重建 | Mip-NeRF 360 `garden` 多视角数据 | `fuse_unbounded_post.ply` | 统一背景环境 |
| Object A | COLMAP + 2DGS | 手机拍摄多视角照片 | `fuse_post.ply` | 真实物体重建 |
| Object B | threestudio / DreamFusion-SD | 文本 prompt | OBJ / MTL / texture | 文本生成虚拟物体 |
| Object C | Magic123 | 单张手机照片 | OBJ / MTL / texture | 单图生成真实物体 |

### 1.1 Garden 背景数据

背景使用 Mip-NeRF 360 数据集中的 `garden` 场景。该数据集提供面向无界真实场景的新视角合成基准，包含高分辨率多视角图像、相机参数和 COLMAP 稀疏重建结果。数据目录结构为：

```text
dataset/360_v2/garden/
├── images/
├── images_2/
├── images_4/
├── images_8/
└── sparse/
```

实验脚本默认读取 `dataset/360_v2/garden`，训练结果保存到 `dataset/garden_2dgs_output`。

### 1.2 Object A 多视角数据与前景预处理

Object A 使用手机环绕拍摄的真实物体照片。完整实验使用 44 张前景图，额外构建了 50% 和 25% 数据量子集，用于分析视角数量对 COLMAP 注册和 2DGS 重建质量的影响。

原始多视角图像包含真实拍摄环境、地面阴影和背景结构。为了避免 2DGS 将背景训练进物体模型，我们使用 `scripts/preprocess_image.py` 进行前景预处理：先调用 `rembg` 的 BRIA-RMBG 前景分割模型，再保留最大前景连通域，最后输出带透明 alpha 通道的 RGBA PNG。2DGS 训练代码通过 patch 读取 alpha mask，将监督图像按

```text
gt_image = image * alpha + background * (1 - alpha)
```

合成，并额外使用 `--lambda_mask` 约束渲染透明度。这样透明区域下方 RGB 不会直接参与训练，减少黑色碎片、白色背景片和阴影残留。

![Object A 原始多视角样例](figures/final1_objectA_raw_grid.jpg)

![Object A 前景预处理结果](figures/final1_objectA_foreground_grid.jpg)

### 1.3 Object B 文本资产

Object B 不需要图像输入，使用 DreamFusion-SD 从文本 prompt 生成蘑菇形虚拟物体。最终 prompt 为：

```text
a zoomed out DSLR product photo of a single umbrella-shaped mushroom, one broad beige brown cap with subtle radial grooves, visible gills under the cap, one straight thick white stem with a small ring, full object, centered, clean silhouette, natural matte surface, detailed organic texture, plain background
```

negative prompt 为：

```text
extra cap, extra stem, stacked body, multiple mushrooms, deformed, distorted, broken geometry, floating parts, messy background, cropped, blurry, text, watermark
```

prompt 设计重点是限制“单个物体、单个菌盖、单根菌柄、完整轮廓”，以减少文本到 3D 常见的多主体、悬浮碎片和几何重复问题。

### 1.4 Object C 单图数据与 Magic123 输入

Object C 使用一张真实兰花照片作为输入。预处理分两步：首先使用自定义 `scripts/preprocess_image.py` 得到带 soft alpha 的 `0001_foreground.png`；然后调用 Magic123 官方 `preprocess_image.py`，复用 alpha 生成 `rgba.png`，并估计 `depth.png`。与 Object A 不同，Object C 保留所有前景连通块，避免花朵、细枝等细结构被最大连通域过滤误删。

![Object C 输入与预处理结果](figures/final1_objectC_preprocess_grid.jpg)

## 2. 方法

### 2.1 COLMAP + 2D Gaussian Splatting

COLMAP 负责从多视角图像中提取 SIFT 特征、进行特征匹配，并通过 Structure-from-Motion 估计相机位姿和稀疏点云。本实验中 Object A 使用 `PINHOLE` 相机模型；这是因为当前 2DGS 数据读取代码支持 `PINHOLE` / `SIMPLE_PINHOLE`，而不直接读取带径向畸变参数的 `SIMPLE_RADIAL`。

2DGS 使用二维高斯面片表示场景表面，相比普通 3D Gaussian 更强调几何表面一致性，并在训练中使用 normal consistency 等正则项。Object A 与 Garden 均使用 2DGS：前者从 COLMAP 重建得到的 `sparse/0` 初始化，后者直接使用 Mip-NeRF 360 数据中已有的 COLMAP 结构。训练后使用 2DGS 的 render/export 流程导出 PLY mesh 代理，供 Blender 融合使用。

### 2.2 DreamFusion / SDS 文本到 3D

Object B 使用 threestudio 中的 DreamFusion-SD 配置。DreamFusion 的核心思想是用预训练 2D 扩散模型提供 Score Distillation Sampling（SDS）梯度，将文本 prompt 的语义约束传递到可微 3D 表示中。训练不需要真实图像，只需要 prompt 和扩散模型先验。

本实验使用 Stable Diffusion v1.5 作为 2D prior，并降低 guidance scale、增加 sparsity loss，以抑制过强 SDS 导致的畸形、底部雾状体积和悬浮结构。训练完成后，通过 threestudio 的 export 阶段将隐式 3D 表示导出为带纹理的 mesh。

### 2.3 Magic123 单图到 3D

Object C 使用 Magic123。Magic123 同时利用 2D 扩散先验和 Zero123 3D 视角先验：2D prior 提供文本语义和纹理想象能力，3D prior 提供从单图预测新视角的一致性约束。实验使用论文推荐的平衡点 `lambda_2d_3d = 1.0`。训练分为 coarse 和 fine 两阶段：coarse 阶段优化隐式表示并导出粗 mesh；fine 阶段从 coarse checkpoint 初始化 DMTet mesh，进一步优化并导出最终 OBJ、MTL 和纹理。

### 2.4 统一表达与 Blender 融合

任务中的表达形式并不一致：2DGS 原生表示是显式高斯面片，DreamFusion 和 Magic123 最终更适合导出 mesh。为了降低融合复杂度，本实验没有进行代码级 Gaussian 拼接，而是统一转换为 Blender 可处理的显式资产：

| 来源 | 原始表达 | 融合表达 | 说明 |
|---|---|---|---|
| Garden | 2D Gaussian / PLY | `fuse_unbounded_post.ply` | 作为背景 mesh 代理，只使用局部可视区域 |
| Object A | 2D Gaussian / PLY | `fuse_post.ply` | 使用最大连通组件过滤小碎片 |
| Object B | 隐式 3D 表示 | OBJ / MTL / texture | threestudio export |
| Object C | NeRF / DMTet | OBJ / MTL / texture | Magic123 fine export |

在 Blender 中，我们手动调整四类资产的尺度、旋转、平移和材质，并设置统一相机轨迹输出漫游视频。PLY 文件导入 Blender 后若颜色显示异常，则使用 `scripts/blender_color.py` 将顶点颜色属性连接到材质 Base Color。

## 3. 实验设置

### 3.1 环境与运行配置

实验统一使用 `cvpj1` conda 环境，核心依赖为 Python 3.8、PyTorch 2.0.1 和 CUDA 11.8。服务器 GPU 为 RTX A6000，因此 CUDA 扩展编译设置 `TORCH_CUDA_ARCH_LIST=8.6` 和 `TCNN_CUDA_ARCHITECTURES=86`。

| 模块 | 入口脚本 | 关键设置 | 输出目录 |
|---|---|---|---|
| Object A | `scripts/run_objectA_colmap_2dgs.sh` | 15000 steps, `lambda_mask=0.1`, eval every 1000 steps | `dataset/object_A*/2dgs_output` |
| Garden | `scripts/run_background_2dgs.sh` | 15000 steps, unbounded render, `depth_ratio=0` | `dataset/garden_2dgs_output` |
| Object B | `scripts/run_objectB_threestudio.sh` | 10000 steps in current script; final available run为 12000 steps | `dataset/object_B*/save` |
| Object C | `scripts/run_objectC_magic123.sh` | coarse 5000 steps + fine 5000 steps, `lambda_2d_3d=1.0` | `dataset/object_C*/magic123_output` |

所有脚本使用 `scripts/timing.sh` 记录阶段耗时，并在训练后通过 `scripts/log_metrics_to_wandb.py` 上传核心指标。W&B 只保留报告需要的少量指标：

| 方法 | W&B 指标 |
|---|---|
| 2DGS | `twodgs/loss_total`, `twodgs/loss_normal`, `twodgs/eval_psnr`, `twodgs/eval_l1_loss` |
| DreamFusion | `dreamfusion/loss_sds`, `dreamfusion/loss_opaque`, `dreamfusion/loss_orient`, `dreamfusion/loss_sparsity` |
| Magic123 | `magic123/loss_total`, `magic123/loss_rgb`, `magic123/loss_mask` |

> 图占位：请从 W&B 导出 `twodgs/loss_total`、`twodgs/loss_normal`、`twodgs/eval_psnr`、`twodgs/eval_l1_loss` 四张图，保存为 `figures/final1_wandb_2dgs_curves.png` 或分别保存为四张子图。

> 图占位：请从 W&B 导出 `dreamfusion/loss_sds`、`dreamfusion/loss_opaque`、`dreamfusion/loss_orient`、`dreamfusion/loss_sparsity` 四张图，保存为 `figures/final1_wandb_dreamfusion_curves.png`。

### 3.2 计算耗时

表 4 来自 `report/timing.csv`。耗时包含预处理、训练和导出阶段，但不包含手动 Blender 融合时间。

| 资产 / 场景 | 阶段 | 耗时（秒） | 耗时（分钟） |
|---|---:|---:|---:|
| Garden | 2DGS train | 3281 | 54.7 |
| Garden | render / export | 713 | 11.9 |
| Object A 100% | preprocess + COLMAP | 645 | 10.8 |
| Object A 100% | 2DGS train | 1398 | 23.3 |
| Object A 100% | render / export | 47 | 0.8 |
| Object A 50% | full pipeline | 1608 | 26.8 |
| Object A 25% | full pipeline | 1397 | 23.3 |
| Object B | DreamFusion train | 4082 | 68.0 |
| Object B | export | 42 | 0.7 |
| Object C | preprocess + Magic123 preprocess | 36 | 0.6 |
| Object C | Magic123 coarse | 3069 | 51.2 |
| Object C | Magic123 fine / export | 1951 | 32.5 |

Object C 总耗时最高，主要来自 coarse 与 fine 两阶段扩散先验优化；Object B 的训练时间也较长，因为每一步都需要 SDS 指导；Object A 的 2DGS 训练相对更快，但前期需要额外进行前景分割和 COLMAP 位姿估计。

## 4. 实验结果

### 4.1 Object A：视角数量与真实物体重建

COLMAP 统计显示，完整 44 张图时注册了 41 张，稀疏点数为 5335，平均重投影误差为 0.8397 px；当数据降到 50% 时，注册图像降到 16 张，点数为 1231；当数据降到 25% 时，仅注册 4 张，点数为 251。该结果说明，物体级环绕重建对视角连续性高度敏感：即使每张图本身清晰，视角覆盖不足也会导致 SfM 图断裂，进而影响 2DGS 初始化和最终几何。

| Object A 数据量 | 输入图像数 | COLMAP 注册图像 | 稀疏点数 | 平均重投影误差 |
|---|---:|---:|---:|---:|
| 100% | 44 | 41 | 5335 | 0.8397 px |
| 50% | 22 | 16 | 1231 | 0.7829 px |
| 25% | 11 | 4 | 251 | 0.7384 px |

注意，较低重投影误差并不代表重建更好。25% 数据只注册了 4 张图，优化问题更小，因此误差可以偏低，但空间覆盖明显不足，无法形成完整物体几何。报告中更应同时关注注册图像数、稀疏点数和最终渲染质量。

![Object A 2DGS 渲染样例](figures/final1_objectA_render_sample.png)

Object A 的前景 alpha 修正是本实验中的关键工程步骤。早期结果中曾出现黑色碎片和背景片状结构；排查后确认，如果训练代码只读取 RGB 而不使用 alpha，透明区域下方的 RGB 会作为普通监督进入训练。最终 patch 后，dataloader 保留 alpha mask，训练阶段用 alpha 合成监督图像，并加入 `lambda_mask=0.1` 的透明度约束，明显降低了背景碎片风险。

### 4.2 Garden：真实背景重建

Garden 作为无界真实场景，适合提供最终融合所需的复杂背景和地面尺度参考。2DGS 可以较好恢复训练视角附近的颜色和几何代理，但导出的 `fuse_unbounded_post.ply` 不是完整高质量三维扫描；它主要服务于 Blender 中的局部视觉背景。对于远处树叶、细草和边界区域，mesh 代理可能出现拉伸和薄片，这是无界场景 mesh 提取的常见现象。

![Garden 2DGS 渲染样例](figures/final1_garden_render_sample.png)

### 4.3 Object B：文本到 3D 生成

Object B 通过文本 prompt 生成蘑菇形资产。最终结果在语义上稳定：能形成单个菌盖和菌柄，整体轮廓清楚；但与真实多视角重建相比，局部几何更依赖扩散先验，容易在底部或轮廓处出现密度扩散、悬浮小块和不合理连接。我们通过降低 guidance scale、增加 sparsity loss、加入 negative prompt 来减少这些问题。

![Object B 生成结果示意](figures/final1_objectB_result.png)

DreamFusion 的四类 W&B 指标用于解释生成稳定性：`loss_sds` 是文本到 3D 的主优化信号，通常震荡明显；`loss_sparsity` 用于鼓励稀疏密度，和悬浮物、糊状体积相关；`loss_orient` 约束表面法线方向；`loss_opaque` 约束不透明度，帮助形成更稳定的实体形状。当前 CSV 日志中最终 step 的指标为：`loss_sds=305.085`、`loss_opaque=0.0262`、`loss_orient=0.00745`、`loss_sparsity=0.488`。这些数值本身不适合和 2DGS 的 photometric loss 横向比较，更适合观察同一训练内部的趋势。

### 4.4 Object C：单图到 3D 生成

Object C 的优势是能够强约束输入视角外观。Magic123 的 `loss_rgb` 和 `loss_mask` 保证已知视图尽量贴合输入图片，Zero123 和 Stable Diffusion prior 则负责补全未知视角。最终 fine 阶段导出 OBJ、MTL 和 `albedo.png`，可以直接进入 Blender。

![Object C 纹理图示意](figures/final1_objectC_albedo.png)

Object C 的主要问题来自单图任务本身的不适定性：背面、遮挡区域和细枝连接关系没有真实多视角约束，只能由先验补全。对于兰花这类细结构物体，花枝和花瓣在 3D 中容易断裂或粘连；因此脚本中保留 soft alpha、提高 `lambda_mask`，并在 fine 阶段设置较高的 known-view 约束频率，以尽量保留输入视图中可见的细枝关系。

下图为 Magic123 coarse / fine 的本地 raw + EMA 曲线。EMA 仅用于报告展示，不上传 W&B，也不改变训练逻辑。

![Object C total loss](figures/final1_objectC_loss_total.png)

![Object C RGB loss](figures/final1_objectC_loss_rgb.png)

![Object C mask loss](figures/final1_objectC_loss_mask.png)

### 4.5 场景融合结果

最终融合使用 Blender 完成。我们将 Garden、Object A、Object B 和 Object C 放入同一坐标空间中，通过人工调整尺度和位置使三类物体在 Garden 地面上具有合理相对大小。最终输出为 `merge.mp4`。

视频占位：

```text
figures/final1_merge.mp4
```

静态图占位：

```text
figures/final1_merge_frame.jpg
```

> 需要补充：最终 LaTeX 报告建议从 `final1_merge.mp4` 截取 1 到 2 帧，保存为 `figures/final1_merge_frame.jpg`，用于正文展示融合结果。

## 5. 对比分析

### 5.1 三种物体生成方式对比

| 维度 | Object A：多视角重建 | Object B：文本到 3D | Object C：单图到 3D |
|---|---|---|---|
| 输入成本 | 高，需要多视角拍摄 | 低，只需要 prompt | 中，只需要单图但需干净前景 |
| 几何准确度 | 最高，受视角数量和 COLMAP 影响 | 最弱，依赖 SDS 和先验想象 | 中等，正面强、背面靠先验 |
| 纹理细节 | 对真实可见区域最好 | 风格化，细节由扩散模型生成 | 输入视图纹理较好，背面不稳定 |
| 可控性 | 由真实数据决定 | prompt 可控但结果随机性大 | 输入图强约束，文本辅助可控 |
| 主要失败模式 | 背景残留、视角不足、反光低纹理 | 悬浮结构、密度糊状、多主体 | 背面幻觉、细结构断裂 |
| 计算耗时 | 中等 | 较高 | 最高 |

Object A 最适合要求真实几何和真实纹理的物体，但需要拍摄质量和视角覆盖；Object B 最适合快速生成语义明确的虚拟资产，但需要 prompt 工程和正则控制；Object C 是单图资产生成的折中方案，适合保留真实输入外观，但对于复杂细结构和不可见背面仍有明显不确定性。

### 5.2 数据量对 Object A 的影响

Object A 的 100% / 50% / 25% 实验说明，减少视角并不会等比例减少训练时间，因为 2DGS 主训练步数固定，真正显著变化的是预处理和 COLMAP 时间；但视角减少会显著降低 COLMAP 注册图像数和稀疏点数，导致几何覆盖不足。因此，真实物体重建中最重要的不是单张图像质量，而是连续视角覆盖、足够重叠和稳定背景控制。

### 5.3 为什么选择 Blender 而不是 Gaussian 级拼接

理论上，可以将 Object B/C mesh 采样为点云，再初始化为 Gaussian primitives，与 Garden 的 2DGS 表达在同一个 renderer 中拼接。但该方案需要处理坐标系、尺度、法线、opacity、颜色、mesh-to-Gaussian 初始化和 renderer 格式兼容，工程复杂度很高。

相比之下，Blender 融合更稳健：2DGS 可导出 PLY mesh 代理，DreamFusion 和 Magic123 可导出 OBJ/MTL/texture，四类资产都能在 Blender 中统一处理。虽然这种方案不保留高斯渲染的原生可微性，但足以满足本任务的资产融合和视频渲染要求。

## 6. 局限性与改进方向

1. Object A 的质量仍依赖前景分割。虽然 alpha mask 已正确参与训练，但如果分割边缘过硬，鞋底、鞋带等细节可能被削弱；如果边缘过软，又可能引入背景残留。后续可尝试实例级视频分割或人工修正 mask。
2. Garden 的 mesh 代理不是完整场景扫描。对于无界场景，2DGS 的 PLY mesh 更适合作为视觉背景，而不是精确几何模型。后续可在 Blender 中只保留局部地面和近景区域。
3. Object B 的 SDS 优化仍有随机性。更复杂的 prompt 或更长训练并不一定提升几何，可能固化错误形状。后续可尝试 ProlificDreamer、MVDream 或多视图扩散先验。
4. Object C 的单图输入无法真实约束背面。对于兰花这类细结构物体，可补充多张参考图，或使用能显式处理多视图一致性的 image-to-3D 方法。
5. 当前融合主要依赖手动 Blender 调整。后续可记录每个资产的变换矩阵和材质设置，提升复现性。

## 7. 结论

本实验完成了一个完整的多源 3D 资产生成与融合流程。真实多视角重建、文本生成和单图生成三条路线在输入成本、几何准确度和纹理可靠性上各有侧重。Object A 证明多视角几何仍然是最可靠的真实物体重建方案；Object B 展示了纯文本生成 3D 资产的灵活性，但也暴露了 SDS 几何不稳定问题；Object C 在单图输入下兼顾了真实外观和未知视角补全，但复杂细结构仍较困难。最终通过 mesh 表达统一和 Blender 融合，我们将三种物体资产插入真实 Garden 背景，完成了任务要求的多视角场景漫游视频。

## 文献

1. Johannes L. Schönberger and Jan-Michael Frahm. *Structure-from-Motion Revisited*. CVPR 2016. COLMAP project: https://colmap.github.io/
2. Bernhard Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. SIGGRAPH 2023. arXiv: https://arxiv.org/abs/2308.04079
3. Binbin Huang et al. *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*. SIGGRAPH 2024. arXiv: https://arxiv.org/abs/2403.17888
4. Jonathan T. Barron et al. *Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields*. CVPR 2022. arXiv: https://arxiv.org/abs/2111.12077
5. Ben Poole et al. *DreamFusion: Text-to-3D using 2D Diffusion*. ICLR 2023. arXiv: https://arxiv.org/abs/2209.14988
6. Robin Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. arXiv: https://arxiv.org/abs/2112.10752
7. Ruoshi Liu et al. *Zero-1-to-3: Zero-shot One Image to 3D Object*. ICCV 2023. arXiv: https://arxiv.org/abs/2303.11328
8. Guocheng Qian et al. *Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors*. arXiv: https://arxiv.org/abs/2306.17843
9. threestudio project. *A Unified Framework for 3D Content Generation*. GitHub: https://github.com/threestudio-project/threestudio
10. BRIA AI. *RMBG Background Removal Model*. Hugging Face: https://huggingface.co/briaai/RMBG-2.0
