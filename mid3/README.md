# Mid3 Oxford-IIIT Pet Segmentation

From `mid3/`, train the three U-Net loss variants:

```bash
python -m src.train --config configs/unet_ce.yaml
python -m src.train --config configs/unet_dice.yaml
python -m src.train --config configs/unet_ce_dice.yaml
```

Evaluate a checkpoint on the fixed validation split:

```bash
python -m src.eval --config configs/unet_ce.yaml --checkpoint outputs/unet_ce/checkpoints/best.pt --split val
```

Evaluation rows are appended to `outputs/eval.csv`. A compact comparison table can be generated with:

```bash
python -m src.summarize
```

Optional prediction visualization:

```bash
python -m src.visualize --config configs/unet_ce.yaml --checkpoint outputs/unet_ce/checkpoints/best.pt --split val
```
