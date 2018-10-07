python train_imagenet.py \
			--ngpu 8 \
			--workers 20 \
			--arch resnet --depth 50 \
			--epochs 100 \
			--batch-size 256 --lr 0.1 \
			--att-type BAM \
			--prefix RESNET50_IMAGENET_BAM \
			./data/ImageNet/
