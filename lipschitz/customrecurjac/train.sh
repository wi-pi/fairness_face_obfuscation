# python train_nlayer.py --model mnist --activation leaky --leaky_slope 0.3 --modelpath models 20 20
python train_nlayer.py --model mnist --activation relu --lr 0.01 --modelpath models 1024 512 256 128 64 32
python train_nlayer.py --model cifar --activation relu --lr 0.001 --wd 0.00002 --epochs 200 --dropout 0.1 --modelpath models 2048 1024 1024 512 512 256 256 128 128
