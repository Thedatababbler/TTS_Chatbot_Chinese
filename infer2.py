
import os
import glob 
import time
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import audio
from utils.utils import *
from synthesizer.taco_synthesizer import *

class Synthesizer():
    """Main entrance for synthesizer"""

    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args
        self.synthesizer = eval(hparams.synthesizer_type)(hparams, args)

    def __call__(self):
        #labels = [fp for fp in glob.glob(
        #    os.path.join(self.args.label_dir, '*'))]
        #for i, label_filename in enumerate(tqdm(labels)):
        label_filename = 'test'
        start = time.time()

        generated_acoustic, acoustic_filename = self.synthesizer(label_filename)
        #if generated_acoustic is None:
        #    print("Ignore {}".format(os.path.basename(label_filename)))
        #    continue
        end = time.time()
        spent = end - start
        n_frame = generated_acoustic.shape[0]
        audio_lens = n_frame * self.hparams.hop_size / self.hparams.sample_rate
        #print("Label: {}, generated wav length: {}, synthesis time: {}, RTF: {}".format(
        #    os.path.basename(label_filename), audio_lens, spent, spent / audio_lens))



def generate_voice(args, flag):

    tf.compat.v1.reset_default_graph()
    hparams = YParams(args.yaml_conf)
    #modified_hp = hparams.parse(args.hparams)
    os.makedirs(args.output_dir, exist_ok=True)
    #if args.alignment_dir is not None:
    #    os.makedirs(args.alignment_dir, exist_ok=True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
    synthesizer = Synthesizer(hparams, args)
    synthesizer()


if __name__ == '__main__':
    main()
