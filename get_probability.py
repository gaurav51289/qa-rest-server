import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
from models.dual_encoder_gru import dual_encoder_model
import csv


tf.flags.DEFINE_string("model_dir", "./runs/GRU", "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "/Users/ektasorathia/Documents/CMPE295B/Final/qa-rest-server/qamodel/runs/GRU/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS
outdir="/Users/ektasorathia/Documents/CMPE295B/udc_train"

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)
# Load vocabulary
#vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
#    160,
#    min_frequency=5,
#    tokenizer_fn=tokenizer_fn)

vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)


QUESTION="""buy 4k laptop would like run ubuntu 1510 believe nt big problem sure adjust text unity ui 
right thing use virtualbox quite often running ubuntu server believe run problem yet set text within guest 
operating system small barely readable right way scaling guest o view like instance scale factor time 2 possible 
run application adapt 4k kind magnifying lens like instance running virtualbox doublesize zoom"""

#ANS=try man telinit however runlevels ubuntu code description 0 halt 1 singleuser mode 2 graphical
# multiuser networking 35 unused configured runlevel 2 6 reboot

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

ans_dict={}

def initialize():
    hparams = udc_hparams.create_hparams()
    model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
    estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)
    estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1, 1]))
    return estimator

def get_probability(context,response,estimator):

    prob = estimator.predict(input_fn=lambda: get_features(context, response))
    probability = next(prob)[0]
    return probability


def read_answers():
    f = open("/Users/ektasorathia/Documents/CMPE295B/udc_train/chatbot-retrieval/answer_clipped.csv", 'rb')
    reader = csv.reader(f)
    for row in reader:
        get_probability(QUESTION,row[0])


if __name__ == "__main__":
   read_answers()