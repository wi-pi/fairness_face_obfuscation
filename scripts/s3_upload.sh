python FaceAPI/s3_upload.py --folder lfw
python FaceAPI/s3_upload.py --folder lfw --model ArcFace --attack lowkey --amplification 1.0 --adversarial-flag true
python FaceAPI/s3_upload.py --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag true
python FaceAPI/s3_upload.py --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute race --different-flag false
python FaceAPI/s3_upload.py --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag true
python FaceAPI/s3_upload.py --folder lfw --model Facenet --attack CW --margin 15.0 --amplification 1.0 --granularity no-amp --adversarial-flag true --attribute sex --different-flag false