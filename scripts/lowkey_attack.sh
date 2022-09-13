python lowkey/attack_dir_warp.py --gpu 0 --model ArcFace --folder lfw --attack lowkey \
--source 0 --attribute race --different-flag false --correct-flag false

python lowkey/attack_dir_warp.py --gpu 0 --model ArcFace --folder lfw --attack lowkey \
--source 1 --attribute race --different-flag false --correct-flag false

python lowkey/attack_dir_warp.py --gpu 1 --model ArcFace --folder lfw --attack lowkey \
--source 2 --attribute race --different-flag false --correct-flag false

python lowkey/attack_dir_warp.py --gpu 1 --model ArcFace --folder lfw --attack lowkey \
--source 3 --attribute race --different-flag false --correct-flag false
