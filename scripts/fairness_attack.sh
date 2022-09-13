python src/attack.py --gpu 0 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 0 --attribute race --different-flag false --correct-flag false
python src/attack.py --gpu 0 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 1 --attribute race --different-flag false --correct-flag false
python src/attack.py --gpu 0 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 2 --attribute race --different-flag false --correct-flag false
python src/attack.py --gpu 2 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 3 --attribute race --different-flag false --correct-flag false

python src/attack.py --gpu 2 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 0 --attribute sex --different-flag true --correct-flag false
python src/attack.py --gpu 2 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 1 --attribute sex --different-flag true --correct-flag false

python src/attack.py --gpu 4 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 0 --attribute race --different-flag true --correct-flag false
python src/attack.py --gpu 4 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 1 --attribute race --different-flag true --correct-flag false
python src/attack.py --gpu 4 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 2 --attribute race --different-flag true --correct-flag false
python src/attack.py --gpu 5 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 3 --attribute race --different-flag true --correct-flag false

python src/attack.py --gpu 5 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 0 --attribute sex --different-flag false --correct-flag false
python src/attack.py --gpu 5 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --source 1 --attribute sex --different-flag false --correct-flag false
