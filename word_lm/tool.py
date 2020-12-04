#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/11/14
Email       : yukunfg@gmail.com
Description : Compute average word length given a text
"""

import argparse
import os
from collections import Counter
import torchtext

def model_param():
    vocab_size = 10000
    emb_dim = 200
    cluster_num = 998
    ngram_num = 5261
    input_params = (vocab_size * 2 + cluster_num + ngram_num) * emb_dim
    bias = vocab_size
    print(f"{(input_params + bias) / (10**6):5.3f}")


def replace_unseen_to_unk(args):
    lm_names = "am en eu id ja zh nl th sk"
    lm_names = lm_names.split()
    lm_dir = "~/pytorch_examples/word_lm/data/50lm/"
    for lm_name in lm_names:
        dir_path = os.path.expanduser(f"{lm_dir}/{lm_name}")
        dir_path = dir_path.rstrip('/')
        dir_prefix, dir_name = os.path.split(dir_path)
        train_path = f"{dir_path}/train.txt"

        test_file = "test.txt"
        valid_file = "valid.txt"
        file_names = [test_file, valid_file]
        

        unk_tag = "<unk>"
        train_counter = Counter()
        lines = [
            train_counter.update(line.strip().split())
            for line in open(train_path, 'r').readlines()
        ]
        new_dir = f"{dir_prefix}/nounseen_{dir_name}"
        os.system(f"mkdir -p {new_dir}")
        os.system(f"cp {train_path} {new_dir}")
        for file_name in file_names:
            file_path = f"{dir_path}/{file_name}"
            file_path_write = f"{new_dir}/{file_name}"
            fh_out = open(file_path_write, "w")
            with open(file_path, 'r') as fh:
                for line in fh:
                    tokens = line.split() 
                    for token_index, token in enumerate(tokens, 0):
                        if token not in train_counter:
                            tokens[token_index] = unk_tag
                    new_line = " ".join(tokens)
                    fh_out.write(f"{new_line}\n")


def get_type_token_ratio(args):
    #  file_path = os.path.expanduser(args.param)
    dir_prefix = "~/common_corpus/data/50lm"
    #  names = "penn wiki2 am ar bg ca cs da de el en es et eu fa fi fr he hi hr hu id it ja jv ka km kn ko lt lv ms mng my nan nl no pl pt ro ru sk sl sr sv ta th tl tr uk vi zh"
    #  names = "kim-cs kim-de kim-es kim-fr kim-ru nounseen_en penn wiki2"
    #  names = "kim-cs kim-de kim-es kim-fr kim-ru nounseen_en penn wiki2"
    #  names = "zh ja en vi th my km"
    names = "zh ja en vi th tl pt my km"
    names = names.split()
    for name in names:
        file_path = f"{dir_prefix}/{name}/train.txt"
        file_path = os.path.expanduser(file_path)
        counter = Counter()
        lines = [
            counter.update(line.strip().split())
            for line in open(file_path, 'r').readlines()
        ]
        token_nums = sum(counter.values())
        vocab_size = len(counter.keys())
        #  print(f"vocab_size: {vocab_size}")
        #  print(f"token_nums: {token_nums}")
        type_token_ratio = vocab_size / token_nums
        #  print(f"type_token_ratio: {type_token_ratio}")
        #  print(f"{name}: {type_token_ratio:5.2f}")
        print(f"{type_token_ratio:5.2f}")
        #  train_path = file_path
        #  test_path = f"{dir_prefix}/{name}/test.txt"
        #  unseen_percent = unseen_word_percentage(train_path, test_path)
        #  unseen_percent = float(unseen_percent)
        #  print(file_path)
        #  print(f"{unseen_percent * 100:.0f}%")


def average_len_word(args):
    dir_prefix = "~/common_corpus/data/50lm"
    #  names = "kim-cs kim-de kim-es kim-fr kim-ru nounseen_en penn wiki2"
    #  names = "zh ja en vi th tl pt my km"
    #  names = "zh vi de en es ar he ja tr"
    #  names = "iwslt15.vi"
    #  names = "kyotofree.ja kyotofree.en"
    names="iwslt15.en-cs.en iwslt15.en-cs.cs iwslt15.en-de.en iwslt15.en-de.de iwslt15.en-fr.en iwslt15.en-fr.fr iwslt15.en-th.en iwslt15.en-th.th iwslt15.en-vi.en iwslt15.en-vi.vi iwslt15.en-zh.en iwslt15.en-zh.zh"
    names = names.split()
    for name in names:
        file_path = f"{dir_prefix}/{name}/train.txt"
        file_path = os.path.expanduser(file_path)
        len_list = []
        with open(file_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                # Skip empty lines
                if line == "":
                    continue
                tokens = line.split()
                for token in tokens:
                    len_list.append(len(token))
        sum_len_list = sum(len_list)
        length = len(len_list)
        print(f"{name}: {sum_len_list / length:5.2f}")
        #  print(f"max length: {max(len_list)}")
        #  print(f"min length: {min(len_list)}")


def unseen_word_percentage(train_path, test_path):
    train_path = os.path.expanduser(train_path)
    test_path = os.path.expanduser(test_path)
    # freqs, itos, stoi
    counter = Counter()
    lines = [
        counter.update(line.strip().split())
        for line in open(train_path, 'r').readlines()
    ]
    train_token_num = sum(counter.values())
    train_vocab = torchtext.vocab.Vocab(counter, specials=[])
    freq_thre = 5
    low_freq_sum = 0
    low_freq_words = set([])
    for item in reversed(counter.most_common()):
        token, freq = item
        if freq >= freq_thre:
            continue
        low_freq_sum += freq 
        low_freq_words.add(token)

    low_freq_percentage = low_freq_sum / train_token_num
    #  print(f"low_freq_percentage in training: {low_freq_percentage}")
    counter = Counter()
    lines = [
        counter.update(line.strip().split())
        for line in open(test_path, 'r').readlines()
    ]
    test_vocab = torchtext.vocab.Vocab(counter, specials=[])

    diff = set(test_vocab.itos) - set(train_vocab.itos)
    #  print(f"unseen words: {len(diff)}")
    unseen_freq = 0
    for unseen in diff:
        unseen_freq += test_vocab.freqs[unseen] 
    all_freq = sum(test_vocab.freqs.values())
    #  print(f"unseen_freq: {unseen_freq}")
    #  print(f"all_freq: {all_freq}")
    #  print(f"percentage: {unseen_freq / all_freq: 5.2f}")
    unseen_percent = f"{unseen_freq / all_freq: 5.2f}"
    return unseen_percent

    low_freq_in_test = 0
    for low_freq_word in low_freq_words:
        if low_freq_word in test_vocab.freqs:
            low_freq_in_test += test_vocab.freqs[low_freq_word]
    low_freq_percentage_in_test = low_freq_in_test / all_freq
    print(f"low_freq_percentage in test: {low_freq_percentage_in_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Description of your program',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-p', '--param',
        help='param',
        required=False
    )
    args = parser.parse_args()
    #  model_param()
    average_len_word(args)
    #  unseen_word_percentage(args)
    #  replace_unseen_to_unk(args)
    #  get_type_token_ratio(args)
