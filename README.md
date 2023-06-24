# STAR
Official implenmentation of SynergisTic pAtch pRuning (STAR)
## Usage
First, clone the repository locally:
```
git clone https://github.com/JuttaZhang/ASTER.git
```
Then, install PyTorch == 1.9.1+cu111\\
prchvision == 0.10.1+cu111 \\
tensorboardx == 2.4\\
tensorboard_logger == 0.1.0:
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install ax-platform == 0.2.4
pip install tensorboardx == 2.4
pip install tensorboard_logger == 0.1.0
```
For example:

Using Bayesian Optimization, VGG-16 and CIFAR-10, run:
```
python BayesianMain.py --model vgg16 --depth 16 --s 1e-4 --exp_flops 0.4 --batch_size 64 --test-batch-size 128 --epochs 320 --pec 0.5 --lb 0.7 --ub 1 
```

For ResNets, run:
```
# using ResNet-56 and CIFAR-100 as an example
python BayesianMaincifar100Res.py --model resnet56 --depth 56 --s 1e-4 --batch_size 64 --test-batch-size 128 --epochs 320 --pec 0.5 --lb 0.1 --ub 1 
```
