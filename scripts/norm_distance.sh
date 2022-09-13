python src/norm_distance.py --gpu 3 --folder lfw --model Facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute race --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model Facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute race
python src/norm_distance.py --gpu 3 --folder lfw --model Facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute sex --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model Facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute sex

python src/norm_distance.py --gpu 2 --folder lfw --model torch_Xu_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model torch_Xu_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race
python src/norm_distance.py --gpu 2 --folder lfw --model torch_Xu_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model torch_Xu_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex

python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_race_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute race --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_race_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute race
python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_race_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute sex --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_race_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute sex

python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_sex_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute race --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_sex_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute race
python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_sex_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute sex --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_sex_balanced_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute sex

python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_default_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute race --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_default_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute race
python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_default_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute sex --different-flag true
python src/norm_distance.py --gpu 2 --folder lfw --model vggface2_default_facenet --attack CW --norm 2 --targeted-flag true \
--hinge-flag true --margin 5.0 --amplification 1.0 --granularity single --adversarial-flag true --attribute sex
