# NODIS
Code of Neural Ordinary Differential Scene Understanding

# Setup
1. Install python and pytorch if you haven't. Our code is based on python 3.6 and pytorch 0.4.1.

2. Path Configuration

2. Compile

3. For a fair comparison we use the pretrained detector checkpoint provided by [MOTIFS](https://github.com/rowanz/neural-motifs). [You can also download it here directly](https://drive.google.com/open?id=1xXIcROgv-u1Yq7ILIyWAndVBQxvP3jUD) and save it under *checkpoints/vgdet/*

4. Train the NODIS model with train_rel.py. We trained the model on a GTX 1080Ti.
+ For PredCLS: *python train_rel.py -m predcls -order random -b 6 -clip 5 -p 100 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/predcls/ -nepoch 20* or [you can download the pretrained NODIS checkpoint here](https://drive.google.com/open?id=1QrhuR3g1I4L_chmMTdNPIr2Zg7-s54nW)
+ For SGCLS: *python train_rel.py -m sgcls -order random -b 6 -clip 5 -p 100 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/predcls/ -nepoch 20*  or [you can download the pretrained NODIS checkpoint here](https://drive.google.com/open?id=1XrPgOiUhcxXZMI_KDMiy4KttzEaO1tWb)
+ For SGDET: *python train_rel.py -m sgdet -order random -b 6 -clip 5 -p 100 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/predcls/ -nepoch 20* or [you can download the pretrained NODIS checkpoint here](https://drive.google.com/open?id=1kOPX7Fj-QW5rMr7HyRgL2h4Tb2RZlCj9)

5. Evaluate the NODIS model with *python eval_rel.py -m predcls -order random -b 6 -clip 5 -p 100 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/sgcls/vgrel-24.tar -nepoch 50*
