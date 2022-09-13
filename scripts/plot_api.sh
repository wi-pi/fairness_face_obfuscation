python src/plot_api_thresholds.py --api-name facepp --folder lfw --model ArcFace --attack lowkey --amplification 1.0 --adversarial-flag true --targeted-success false
python src/plot_api_thresholds.py --api-name azure --folder lfw --model ArcFace --attack lowkey --amplification 1.0 --adversarial-flag true --targeted-success false
python src/plot_api_thresholds.py --api-name awsverify --folder lfw --model ArcFace --attack lowkey --amplification 1.0 --adversarial-flag true --targeted-success false

python src/plot_api_thresholds.py --api-name facepp --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag false --attribute race
python src/plot_api_thresholds.py --api-name azure --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag false --attribute race
python src/plot_api_thresholds.py --api-name awsverify --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag false --attribute race
python src/plot_api_thresholds.py --api-name facepp --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag false --attribute sex
python src/plot_api_thresholds.py --api-name azure --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag false --attribute sex
python src/plot_api_thresholds.py --api-name awsverify --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag false --attribute sex

python src/plot_api_thresholds.py --api-name facepp --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag true --attribute race
python src/plot_api_thresholds.py --api-name azure --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag true --attribute race
python src/plot_api_thresholds.py --api-name awsverify --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag true --attribute race
python src/plot_api_thresholds.py --api-name facepp --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag true --attribute sex
python src/plot_api_thresholds.py --api-name azure --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag true --attribute sex
python src/plot_api_thresholds.py --api-name awsverify --folder lfw --model Facenet --attack CW --amplification 1.0 --adversarial-flag true --targeted-success false --different-flag true --attribute sex
