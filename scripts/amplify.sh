python src/amplify.py --gpu 1 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --attribute race --different-flag false --adversarial-flag true

python src/amplify.py --gpu 1 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --attribute race --different-flag true --adversarial-flag true --source 0 --target 3

python src/amplify.py --gpu 1 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --attribute sex --different-flag true --adversarial-flag true

python src/amplify.py --gpu 1 --model Facenet --folder lfw --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.3 --margin 15.0 --amplification 1.0 \
--iterations 40 --binary-steps 3 --learning-rate 0.01 --epsilon-steps 0.03 --init-const 0.3 --interpolation bilinear \
--granularity no-amp --batch-size 10 --pair-flag true --mean-loss embeddingmean --attribute sex --different-flag true --adversarial-flag true
