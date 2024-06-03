# STAR
The implementation of SynergisTic pAtch pRuning for Vision Transformers.
# This code is the implementation of STAR
[Synergistic Patch Pruning for ViT: Unifying Intra- \& Inter-Layer Patch Importance]

Author: Yuyao Zhang, Lan Wei, and Nikolaos M. Freris

## Usage

First, install PyTorch == 1.9.1+cu111\\
prchvision == 0.10.1+cu111 \\
tensorboardx == 2.4\\
tensorboard_logger == 0.1.0:
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardx == 2.4
pip install tensorboard_logger == 0.1.0
```

# using DeiT-Base and ImageNet as an example
generate the inter-layer important scores for each patch.
```
python LRP_attention_heatmap.py
```
then, use ACS.ipynb to get the retention rates.


Fine-tuning process:
```
nohup python -m torch.distributed.launch --master_port 29500 --nproc_per_node=8 --use_env ./pruning/pruning_py/finetune.py --device cuda --epochs 120 --lr 2e-4 --min-lr 2e-10 --dist-eval --model deit_base_distilled_patch16_224  --batch-size 256 --distillation-alpha 0.25 --distillation-type hard --warmup-epochs 0 --finetune-only --output-dir ... --data-path ... --resume ... >finetune_deit_base_120.log 2>&1 &
```
