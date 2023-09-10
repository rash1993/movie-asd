# download the Keras_vgg_face_2 model
mkdir -p Keras_VGGFace2_ResNet50/weights
mkdir -p Pytorch_Retinaface/weights
gdown 11JQyEfCdo7cqA8IDbUa8JadVK65YG_BA -O Keras_VGGFace2_ResNet50/weights/
gdown 11C_g1bwOWCU13T6Gf2FDVt8AhLKTN64N -O Pytorch_Retinaface/weights/
gdown 1GJU2e8_lb_n81qjVLJZI2pKmgYgokKLm -O voxceleb_trainer/
export PATH=~/.local/bin:$PATH
