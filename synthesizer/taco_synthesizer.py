import os
import argparse
import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.utils import *
from text.feature_converter import *
from synthesizer.basic_synthesizer import BaseSynthesizer
from converter import *
from parse_text_add_pause import *


class TacotronSynthesizer(BaseSynthesizer):
    """Synthesizer for Tacotron model"""

    def __init__(self, hparams, args):
        self.tuple_value = "phones input_length acoustic_targets targets_length stop_token_targets"
        self.text = args.text_input
        self.pause = TTSSegPause()
        BaseSynthesizer.__init__(self, hparams, args)

    def make_test_feature(self):
        phones = tf.placeholder(tf.int32, [1, None], 'phones')
        input_length = tf.placeholder(tf.int32, [1], 'input_length')
        acoustic_targets = None #  tf.placeholder(tf.float32, [1, None, self.hparams.acoustic_dim], 'acoustic_targets')

        return self.test_sample(
            phones=phones,
            input_length=input_length,
            acoustic_targets=acoustic_targets,
            targets_length=None,
            stop_token_targets=None)

    def make_feed_dict(self, text):
        hparams = self.hparams
        new_text = self.pause.add_pause(text)#text_cut(self.text)
        new_text = new_text.replace('，','')
        new_text = new_text.replace('！','')
        new_text = new_text.replace('。','')
        pyin, _ = get_pyin(new_text)
        #print(pyin)
        phones= label_to_sequence2(pyin)
        #print(phones)

        feed_dict = {}
        feed_dict[self.model.phones] = [np.asarray(phones, dtype=np.int32)]
        feed_dict[self.model.input_length] = np.asarray(
            [len(phones)], dtype=np.int32)

        return feed_dict
