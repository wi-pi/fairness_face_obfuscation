python src/new_api_eval.py --api-name facepp --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag false
python src/new_api_eval.py --api-name facepp --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python src/new_api_eval.py --api-name facepp --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag false
python src/new_api_eval.py --api-name facepp --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true

python src/new_api_eval.py --api-name azure --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python src/new_api_eval.py --api-name azure --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true

python src/new_api_eval.py --api-name awsverify --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python src/new_api_eval.py --api-name awsverify --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true

python src/new_api_eval.py --api-name facepp --folder lfw --model ArcFace --attack lowkey --amplification 1.0 --adversarial-flag true --targeted-success false
python src/new_api_eval.py --api-name azure --folder lfw --model ArcFace --attack lowkey --amplification 1.0 --adversarial-flag true --targeted-success false
python src/new_api_eval.py --api-name awsverify --folder lfw --model ArcFace --attack lowkey --amplification 1.0 --adversarial-flag true --targeted-success false