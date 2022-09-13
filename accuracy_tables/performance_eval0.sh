#python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Asian --all-flag false --uniform-flag false
#python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Black --all-flag false --uniform-flag false
#python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute White --all-flag false --uniform-flag false
#python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Indian --all-flag false --uniform-flag false
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Male --all-flag false --uniform-flag false --model resnet18_default_softmax
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute NOTMale --all-flag false --uniform-flag false --model resnet18_default_softmax
#python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Asian --all-flag true --uniform-flag false
#python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Black --all-flag true --uniform-flag false
#python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute White --all-flag true --uniform-flag false
#python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Indian --all-flag true --uniform-flag false
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Male --all-flag true --uniform-flag false --model balance_sex_3
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute NOTMale --all-flag true --uniform-flag false --model balance_sex_3

#python fpfair/facenet_lfw.py --gpu 3 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Male --all-flag true --uniform-flag false --model balance_sex_4
#python fpfair/facenet_lfw.py --gpu 3 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute NOTMale --all-flag true --uniform-flag false --model balance_sex_4

# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Asian --all-flag true --uniform-flag false --model balance_race_3
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Black --all-flag true --uniform-flag false --model balance_race_3
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute White --all-flag true --uniform-flag false --model balance_race_3
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Indian --all-flag true --uniform-flag false --model balance_race_3

# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Male --all-flag true --uniform-flag false --model default_softmax_3
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute NOTMale --all-flag true --uniform-flag false --model default_softmax_3
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Asian --all-flag true --uniform-flag false --model default_softmax_3
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Black --all-flag true --uniform-flag false --model default_softmax_3
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute White --all-flag true --uniform-flag false --model default_3
# python fpfair/facenet_lfw.py --gpu 2 --folder lfw --model-type large --loss-type triplet --dataset-type vgg --attribute Indian --all-flag true --uniform-flag false --model default_3

python fpfair/validate_on_lfw.py fpfair/lfw/lfw-align-160 default --lfw_pairs lfw_pairs_Male.txt --attribute Male --all false --gpu 0
python fpfair/validate_on_lfw.py fpfair/lfw/lfw-align-160 default --lfw_pairs lfw_pairs_NOTMale.txt --attribute NOTMale --all false --gpu 0
python fpfair/validate_on_lfw.py fpfair/lfw/lfw-align-160 default --lfw_pairs lfw_pairs_Asian.txt --attribute Asian --all false --gpu 0
python fpfair/validate_on_lfw.py fpfair/lfw/lfw-align-160 default --lfw_pairs lfw_pairs_Black.txt --attribute Black --all false --gpu 0
python fpfair/validate_on_lfw.py fpfair/lfw/lfw-align-160 default --lfw_pairs lfw_pairs_White.txt --attribute White --all false --gpu 0
python fpfair/validate_on_lfw.py fpfair/lfw/lfw-align-160 default --lfw_pairs lfw_pairs_Indian.txt --attribute Indian --all false --gpu 0

python fpfair/validate_on_lfw.py fpfair/lfw/lfw-align-160 default --lfw_pairs lfw_pairs_all_Male.txt --attribute Male --all true --gpu 0
python fpfair/validate_on_lfw.py fpfair/lfw/lfw-align-160 default --lfw_pairs lfw_pairs_all_NOTMale.txt --attribute NOTMale --all true --gpu 0
python fpfair/validate_on_lfw.py fpfair/lfw/lfw-align-160 default --lfw_pairs lfw_pairs_all_Asian.txt --attribute Asian --all true --gpu 0


