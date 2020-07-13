# NODIS: Neural Ordinary Differential Scene Understanding
Here is the pytorch code for our paper [NODIS: Neural Ordinary Differential Scene Understanding (**ECCV 2020**)](https://arxiv.org/abs/2001.04735v2). If the paper is helpful for you, we request that you cite our work.

## Setup
1. Install python and pytorch if you haven't. Our code is based on python 3.6 and pytorch 0.4.1.

2. Compile

3. For a fair comparison we use the pretrained detector checkpoint provided by [MOTIFS](https://github.com/rowanz/neural-motifs). [You can also download it here directly](https://drive.google.com/open?id=1xXIcROgv-u1Yq7ILIyWAndVBQxvP3jUD) and save it under *checkpoints/vgdet/*

## Training
You can train the NODIS model with train_rel.py. We trained the model on a GTX 1080Ti.
+ For PredCLS: 
```python
python train_rel.py -m predcls -order random -b 6 -clip 5 -p 100 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/predcls/ -nepoch 20
```
+ For SGCLS: 
```python
python train_rel.py -m sgcls -order random -b 6 -clip 5 -p 100 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/predcls/ -nepoch 20
```
+ For SGDET: 
```python
python train_rel.py -m sgdet -order random -b 6 -clip 5 -p 100 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/predcls/ -nepoch 20
```
or [you can download the pretrained NODIS checkpoint here](https://drive.google.com/open?id=1kOPX7Fj-QW5rMr7HyRgL2h4Tb2RZlCj9)

## Evaluation
You can evaluate the NODIS model with 
```python
python eval_rel.py -m predcls -order random -b 6 -clip 5 -p 100 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/sgcls/vgrel-24.tar -nepoch 50
```
