CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_imagenet.py \
			--ngpu 4 \
			--workers 20 \
			--arch resnet --depth 50 \
			--epochs 100 \
			--batch-size 256 --lr 0.1 \
			--att-type BAM \
			--prefix RESNET50_IMAGENET_BAM \
			./data/ImageNet/ \
			> logs/RESNET50_IMAGENET_BAM.log
