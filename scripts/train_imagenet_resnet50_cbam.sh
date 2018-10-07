CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_imagenet.py \
			--ngpu 4 \
			--workers 20 \
			--arch resnet --depth 50 \
			--epochs 100 \
			--batch-size 256 --lr 0.1 \
			--att-type CBAM \
			--prefix RESNET50_IMAGENET_CBAM \
			./data/ImageNet/ \
			> logs/RESNET50_IMAGENET_CBAM.log
