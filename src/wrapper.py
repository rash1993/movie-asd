from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import pdb
import argparse
import utils as ut
import numpy as np


global args
args = {'net': 'resnet50', 'loss': 'softmax', 'aggregation':'avg', 'batch_size':256, \
'resume':'/data/face_clustering/baselines/Keras-VGGFace2-ResNet50/vggface2_Keras/model/resnet50_softmax_dim512/weights.h5',\
'mode': 'eval', 'gpu': '0', 'feature_dim': 512, 'data_path': None, 'save_path': None}
args = Namespace(**args)

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def initialize_model():
    import model
    model_eval = model.Vggface2_ResNet50(mode='eval')
    weights_path = os.path.join('../Keras-VGGFace2-ResNet50/weights/weights.h5')
    model_eval.load_weights(weights_path, by_name=True)

def image_encoding(model, facepaths):
    batch_size = 256
    num_faces = len(facepaths)
    face_feats = np.empty((num_faces, args.feature_dim))
    imgpaths = facepaths.tolist()
    imgchunks = list(chunks(imgpaths, batch_size))

    for c, imgs in enumerate(imgchunks):
        im_array = np.array([ut.load_data(path=i, shape=(224, 224, 3), mode='eval') for i in imgs])
        f = model.predict(im_array, batch_size=args.batch_size)
        start = c * args.batch_size
        end = min((c + 1) * args.batch_size, num_faces)
        face_feats[start:end] = f
        # if c % 500 == 0:
        #     print('-> finish encoding {}/{} images.'.format(c*args.batch_size, num_faces))
    return face_feats

