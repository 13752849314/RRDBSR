# ESRGAN: Enhanced Super-Resolution Generative Adversarial Network

[pdf](https://arxiv.org/pdf/1809.00219v2.pdf)

# Enhanced Deep Residual Networks for Single Image Super-Resolution

[pdf](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-CVPRW-2017.pdf)

## Download by

```
git clone git@github.com:13752849314/RRDBSR.git
```

train data at [here](https://www.kaggle.com/datasets/quadeer15sh/image-super-resolution-from-unsplash)

## quick start

```
python train.py -opt RRDB_train.yml
```

运行前请先配置 *.yml文件

## Result

| Method | Dataset | Scale | PSNR      | SSIM     |
|--------|---------|-------|-----------|----------|
| EDSR   | ISR     | x4    | 27.81     | 0.79     |
| RRDBSR | ISR     | x4    | **30.64** | **0.84** |
| EDSR   | Set5    | x4    | 26.20     | 0.84     |
| RRDBSR | Set5    | x4    | **31.96** | **0.92** |
