import os
import pickle

import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from keras.preprocessing.image import img_to_array
from scipy.misc import imresize

from operator import itemgetter
from json import load

from ai.w2v import get_words, get_words_vector, load_w2v_model

EPSILON = 1e-6

LIKELIHOOD_VALUES = {
        'UNKNOWN':        -1.0,
        'VERY_UNLIKELY':  0.0,
        'UNLIKELY':       0.25,
        'POSSIBLE':       0.5,
        'LIKELY':         0.75,
        'VERY_LIKELY':    1.0,
    }

class MergeModel():
    
    def __init__(self, path):

        self.path = path

        self.model = None
        self.load_model()
        self.load_w2v_model()

    def load_w2v_model(self):
        self.w2v_model = load_w2v_model(os.path.join(self.path,                                                  'w2v_slim.bin.gz'))

    def load_model(self):
        with keras.utils.generic_utils.CustomObjectScope({
            'relu6': keras.applications.mobilenet.relu6,                                                          
            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
                }):
            simple_model_path = os.path.join(self.path, 'simple/keras_model.h5')
            print('Loading model {}'.format(simple_model_path))
            self.simple_model = keras.models.load_model(simple_model_path)
            self.simple_graph = tf.get_default_graph()
            self.simple_model._make_predict_function()

            cnn_model_path = os.path.join(self.path, 'cnn/model.h5')
            print('Loading model {}'.format(cnn_model_path))
            self.cnn_model = keras.models.load_model(cnn_model_path)
            self.cnn_graph = tf.get_default_graph()
            self.cnn_model._make_predict_function()

            merge_model_path = os.path.join(self.path, 'merge/keras_model.h5')
            print('Loading model {}'.format(merge_model_path))
            self.merge_model = keras.models.load_model(merge_model_path)
            self.merge_graph = tf.get_default_graph()
            self.merge_model._make_predict_function()


    def run(self, image, features, stats):

        print('Running: {}'.format(self.path))
        with open(os.path.join(self.path, 'tags.json'), 'r') as f:
            tags = load(f)

        scaler_X = pickle.load(open(os.path.join(self.path, 'simple/scaler_X.pkl'), 'rb'))

        parsed_features = self.parse_image_features(features, stats, tags)
        sorted_features = [parsed_features[k] for k
                           in sorted(parsed_features.keys())]

        tag_columns = [i for i in sorted(parsed_features.keys())
                       if 'tag' == i[0:3]]

        non_tag_columns = [i for i in sorted(parsed_features.keys())
                           if 'tag' != i[0:3]]

        words = get_words(pd.DataFrame([parsed_features]), tag_columns)
        word_vector = get_words_vector(words, model=self.w2v_model)

        sorted_features = [parsed_features[k] for k in non_tag_columns] \
            + word_vector.tolist()

        X = np.array([sorted_features])
        X_scaled = scaler_X.transform(X)
        assert(
            (np.abs(scaler_X.inverse_transform(X_scaled) - X) < EPSILON).all()
        )

        target_shape = (160, 160)
        img_arr = img_to_array(image)
        img_arr = imresize(img_arr, target_shape)
        img_arr = img_arr / 255.

        with self.cnn_graph.as_default():
            y_img = self.cnn_model.predict(np.array([img_arr]))

        with self.simple_graph.as_default():
            y_simple = self.simple_model.predict(X_scaled)

        x_merge = np.concatenate([y_simple, y_img], axis=1)

        with self.merge_graph.as_default():
            Y = self.merge_model.predict(x_merge)

        return Y[0]

    def parse_image_features(self, cloudvision, stats, tags):

        result = {}
        labels = cloudvision.get('labelAnnotations', [])

        for tag in tags:
            result['tag_' + tag.replace(' ', '_')] = 0.0

        for label in labels:
            tag_name = 'tag_' + label['description'].replace(' ', '_')
            if tag_name in result:
                result[tag_name] = label['score']

        result['num_tags'] = len(labels)

        for k, v in stats.items():
            result[k] = v

        for k, v in cloudvision.get('safeSearchAnnotation', {}).items():

            if k in ('adult', 'medical', 'spoof', 'violence', 'racy'):
                result[k] = LIKELIHOOD_VALUES[v]

        faces = cloudvision.get('faceAnnotations', [])
        result['num_faces'] = len(faces)

        for field in (('angerLikelihood',
                    'joyLikelihood',
                    'sorrowLikelihood',
                    'surpriseLikelihood')):

            values = [LIKELIHOOD_VALUES[face[field]] for face in faces]
            field_name = field.replace('Likelihood', '')
            result[field_name] = sum(values) / (len(values) if values else 1)
            result[field_name + '_max'] = max(values) if values else 0.0

        image_colors = cloudvision['imagePropertiesAnnotation']['dominantColors']['colors']
        image_colors.sort(key=itemgetter('score'), reverse=True)

        for i in range(3):
            result['color_{}_red'.format(i)] = -1.0
            result['color_{}_green'.format(i)] = -1.0
            result['color_{}_blue'.format(i)] = -1.0

        for i, color in enumerate(image_colors[:3]):
            result['color_{}_red'.format(i)] = color['color']['red']
            result['color_{}_green'.format(i)] = color['color']['green']
            result['color_{}_blue'.format(i)] = color['color']['blue']

        result['num_landmarks'] = len(cloudvision.get('landmarkAnnotations', []))

        return result
