import sys
import os
import udc_model
import udc_hparams
import dual_encoder_gru
import numpy as np
import tensorflow as tf
from flask_api import FlaskAPI
from flask import request, jsonify
from flask_cors import CORS

tf.flags.DEFINE_string("model_dir", "models/GRU", "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "models/GRU/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not os.path.exists(FLAGS.model_dir):
    print("You must specify a model directory")
    sys.exit(1)


if not os.path.isfile(FLAGS.vocab_processor_file):
    print("You must specify a vocab_processor_file")
    sys.exit(1)


def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)

vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_processor_file)

def get_features(context, utterance):
    context_matrix = np.array(list(vp.transform([context])))
    utterance_matrix = np.array(list(vp.transform([utterance])))
    context_len = len(context.split(" "))
    utterance_len = len(utterance.split(" "))
    features = {
        "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
        "context_len": tf.constant(context_len, shape=[1, 1], dtype=tf.int64),
        "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
        "utterance_len": tf.constant(utterance_len, shape=[1, 1], dtype=tf.int64),
    }
    return features, None


def gpinitialize():
    hparams = udc_hparams.create_hparams()
    model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_gru.dual_encoder_model)
    estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)
    estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1, 1]))
    return estimator


def get_probability(context, response, estimator):
    prob = estimator.predict(input_fn=lambda: get_features(context, response))
    probability = next(prob)[0]
    return probability

estimator = gpinitialize()

def create_app():
    app = FlaskAPI(__name__, instance_relative_config=True)

    CORS(app)

    @app.route('/qa/', methods=['POST'])
    def ask():
        if request.method == "POST":
            question = str(request.data.get('question'))
            candidate = str(request.data.get('candidate'))

            # TODO: answer id in the request probably

            if question and candidate:
                # TODO: calculate probability

                probability = get_probability(question, candidate, estimator)
                # probability = 0.456

                print(probability)
                response = jsonify({
                    'probability': str(probability)
                })

                return response
    return app


app = create_app()

if __name__ == '__main__':
    app.run(port=5002)
