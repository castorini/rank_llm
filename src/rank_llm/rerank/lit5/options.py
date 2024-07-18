import argparse
import os
from pathlib import Path

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_eval_options(self):
        self.parser.add_argument('--write_crossattention_scores', action='store_true',
                                 help='take relevance cross-attention scores from model')
        self.parser.add_argument('--bfloat16', action='store_true',
                                 help='Run model inference in bfloat16')
        self.parser.add_argument('--stride', type=int, default=1)
        self.parser.add_argument('--n_rerank_passages', type=int, default=1)
        self.parser.add_argument('--sort_key', type=str, default='none')
        self.parser.add_argument('--n_passes', type=int, default=1)

    def add_reader_options(self):
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--text_maxlength', type=int, default=150,
                                 help='maximum number of tokens in text segments (query+passage)')
        self.parser.add_argument('--answer_maxlength', type=int, default=-1,
                                 help='maximum number of tokens to generate')
        self.parser.add_argument('--n_passages', type=int, default=1)


    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--runfile_path', type=str, help='.trec runfiles are saved here')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for model')

        # dataset parameters
        self.parser.add_argument("--batch_size", default=1, type=int,
                                 help="Batch size per GPU/CPU")


    def parse(self):
        opt = self.parser.parse_args()
        return opt