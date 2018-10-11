# BAM and CBAM
Official PyTorch code for "BAM: Bottleneck Attention Module (BMVC2018)" and "CBAM: Convolutional Block Attention Module (ECCV2018)"

### Updates & Notices
- 2018-10-08: ~~Currently, only CBAM test code is validated. **There may be minor errors in the training code**. Will be fixed in a few days.~~
- 2018-10-11: Training code validated. RESNET50+BAM pretrained weight added.

### Requirement

The code is validated under below environment:
- Ubuntu 16.04, 4 GTX 1080 Ti, Docker (PyTorch 0.4.1, CUDA 9.0 + CuDNN 7.0, Python 3.6)

### How to use

ResNet50 based examples are included. Example scripts are included under ```./scripts/``` directory.
ImageNet data should be included under ```./data/ImageNet/``` with foler named ```train``` and ```val```.

```
# To train with BAM (ResNet50 backbone)
python train_imagenet.py --ngpu 4 --workers 20 --arch resnet --depth 50 --epochs 100 --batch-size 256 --lr 0.1 --att-type BAM --prefix RESNET50_IMAGENET_BAM ./data/ImageNet
# To train with CBAM (ResNet50 backbone)
python train_imagenet.py --ngpu 4 --workers 20 --arch resnet --depth 50 --epochs 100 --batch-size 256 --lr 0.1 --att-type CBAM --prefix RESNET50_IMAGENET_CBAM ./data/ImageNet
```

### Resume with checkpoints

- ResNet50+CBAM (trained for 100 epochs) checkpoint is provided in this [link](https://www.dropbox.com/s/bt6zty02h9ibufi/RESNET50_CBAM_new_name_wrap.pth?dl=0). ACC@1=77.622 ACC@5=93.948
- ResNet50+BAM (trained for 90 epochs) checkpoint is provided in this [link](https://www.dropbox.com/s/esw0m8e3cjg7ex4/RESNET50_IMAGENET_BAM_best.pth.tar?dl=0). ACC@1=76.860 ACC@5=93.416

For validation, please use the script as follows
```
python train_imagenet.py --ngpu 4 --workers 20 --arch resnet --depth 50 --att-type CBAM --prefix EVAL --resume $CHECKPOINT_PATH$ --evaluate ./data/ImageNet
```
