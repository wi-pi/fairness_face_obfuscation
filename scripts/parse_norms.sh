python src/parse_norms.py --gpu 3 --folder lfw --model Facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python src/parse_norms.py --gpu 3 --folder lfw --model Facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true

python src/parse_norms.py --gpu 2 --folder lfw --model torch_Xu_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python src/parse_norms.py --gpu 2 --folder lfw --model torch_Xu_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true

python src/parse_norms.py --gpu 2 --folder lfw --model vggface2_race_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python src/parse_norms.py --gpu 2 --folder lfw --model vggface2_race_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true

python src/parse_norms.py --gpu 2 --folder lfw --model vggface2_sex_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python src/parse_norms.py --gpu 2 --folder lfw --model vggface2_sex_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true

python src/parse_norms.py --gpu 2 --folder lfw --model vggface2_default_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python src/parse_norms.py --gpu 2 --folder lfw --model vggface2_default_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true
