# NODIS: Neural Ordinary Differential Scene Understanding
Here is the pytorch code for our paper [NODIS: Neural Ordinary Differential Scene Understanding (**ECCV 2020**)](https://arxiv.org/abs/2001.04735v2). If the paper is helpful for you, we request that you cite our work.

![GitHub Logo](/docs/teaser_eccv.png)

**Our code is supported by [neural-motifs](https://github.com/rowanz/neural-motifs) and [torchdiffeq](https://github.com/rtqichen/torchdiffeq). Great thanks to [Rowan Zellers](https://github.com/rowanz) and [Ricky Chen](https://github.com/rtqichen)!**

## Setup
1. Install python and pytorch if you haven't. Our code is based on python 3.6 and pytorch 0.4.1.

2. Compile: run ```make``` in the main directory

3. Download Neural ODE module [here](https://github.com/rtqichen/torchdiffeq/tree/master/torchdiffeq)

4. For a fair comparison we use the pretrained object detector checkpoint provided by [neural-motifs](https://github.com/rowanz/neural-motifs). [You can download it here directly](https://drive.google.com/open?id=1xXIcROgv-u1Yq7ILIyWAndVBQxvP3jUD) and save it under *checkpoints/vgdet/*

5. The final directories for data and detection models should look like:
```
|-- checkpoints
|   |-- vgdet
|-- data
|   |-- stanford_filtered
|-- dataloaders
|-- lib
|-- torchdiffeq
|-- models
```

## Training
You can train the NODIS model with train_rel.py. We trained the model on a GTX 1080Ti.
+ For PredCLS: 
```python
python train_rels.py -m predcls -order random -b 6 -p 100 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/ -nepoch 20
```
+ For SGCLS: 
```python
python train_rels.py -m sgcls -order random -b 6 -p 100 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/ -nepoch 20
```
+ For SGGEN: 
```python
python train_rels.py -m sgdet -order random -b 6 -p 100 -lr 1e-4 -ngpu 1 -ckpt $CHECKPOINT -save_dir checkpoints/ -nepoch 20
```


## Evaluation
You can evaluate the model trained by yourself.
+ For PredCLS: 
```python
python eval_rels.py -m predcls -order random -b 6 -p 100 -lr 1e-3 -ngpu 1 -test -ckpt $CHECKPOINT -nepoch 50
```
+ For SGCLS: 
```python
python eval_rels.py -m sgcls -order random -b 6 -p 100 -lr 1e-3 -ngpu 1 -test -ckpt $CHECKPOINT -nepoch 50
```
+ For SGGEN: 
```python
python eval_rels.py -m sgdet -order random -b 6 -p 100 -lr 1e-3 -ngpu 1 -test -ckpt $CHECKPOINT -nepoch 50
```
or you can download the pretrained NODIS PREDCLS/[SGCLS](https://drive.google.com/file/d/1CnZpAas29aayQLDBf5brQkIIexI952gI/view?usp=sharing)/[SGGEN](https://drive.google.com/file/d/1cSj4SX80P5B8wb-5bISJJKCCVziIUqvQ/view?usp=sharing) here.

## Help
This is a draft version, if you find any problem, please contact with us.
