import argparse
import sys

sys.path.append("../word_lm/")
from word_emb import load_word_emb

WORDSIM_DATA_PATHS = [
    "/home/lr/yukun/common_corpus/data/embedding-evaluation/wordsim/data/en/EN-WS-353-ALL.txt",
    "/home/lr/yukun/common_corpus/data/embedding-evaluation/wordsim/data/en/EN-RW-STANFORD.txt",
    "/home/lr/yukun/common_corpus/data/embedding-evaluation/wordsim/data/en/EN-MEN-TR-3k.tab.txt",
    "/home/lr/yukun/common_corpus/data/embedding-evaluation/wordsim/data/en/EN-MTurk-771.tab.txt"
    ]

ANALOGY_DATA_PATHS = [
    "/home/lr/yukun/common_corpus/data/embedding-evaluation/analogy/en/questions-words.txt"
    ]


def load_words_from_wordsim(task_data_path, word_set):
  with open(task_data_path, 'r') as fh:
    for line in fh:
      line = line.strip()
      if line == "":
        continue
      try:
        word1, word2, score = line.split('\t')
      except Exception as e:
        raise Exception(f"{line} in {task_data_path} is not valid for wordsim")
      word_set.add(word1)
      word_set.add(word2)
  return word_set


def load_words_from_analogy(task_data_path, word_set):
  with open(task_data_path, 'r') as fh:
    for line in fh:
      line = line.strip()
      if line == "" or line.startswith(":"):
        continue
      words = line.split(' ')
      if len(words) != 4:
        raise Exception(f"{line} is not valid line for analogy data.")
      for word in words:
        word_set.add(word)
  return word_set


def use_fasttext_computed_model(args, model):
  # Load words that will be used in evaluation and compute their embs.
  word_list = set()
  for task_data_path in WORDSIM_DATA_PATHS:
    load_words_from_wordsim(task_data_path, word_list)
  for task_data_path in ANALOGY_DATA_PATHS:
    load_words_from_analogy(task_data_path, word_list)
    word_list = list(word_list)

  tmp_txt_model = f"tmp_word_emb.txt"
  unseen = []
  word_emb_strs = []
  for word in word_list:
    if word in model:
      word_emb = model[word]
      word_emb_str = " ".join(word_emb.astype(str))
      word_emb_str = f"{word} {word_emb_str}"
      word_emb_strs.append(word_emb_str)
    else:
      unseen.append(word)
  print(f"Unsee words: {len(unseen)}/{len(word_list)}, some are: {unseen[0:20]}")

  with open(tmp_txt_model, 'w') as fh:
    fh.write(f"{len(word_emb_strs)} {model.vector_size}\n")
    for word_emb_str in word_emb_strs:
      fh.write(f"{word_emb_str}\n")
  from gensim.models import KeyedVectors
  model = KeyedVectors.load_word2vec_format(tmp_txt_model, binary=False)
  import os
  os.system(f"rm -r {tmp_txt_model}")
  return model

def emb_eval(args):
  model = load_word_emb(args.input_emb_path, args.emb_format)

  # For fasttext model, we compute word representation on test set first.
  if args.emb_format == "fasttext":
    model = use_fasttext_computed_model(args, model)

  # Eval on word similarity tasks:
  for task_data_path in WORDSIM_DATA_PATHS:
    similarities = model.wv.evaluate_word_pairs(task_data_path)
    print(f"{task_data_path}: \n{similarities}")

  for task_data_path in ANALOGY_DATA_PATHS:
    #  analogy_scores = model.wv.accuracy(task_data_path)
    analogy_scores = model.wv.evaluate_word_analogies(task_data_path)
    print(f"{task_data_path}: \n{analogy_scores[0]}")
  print(f"Finishing evaluating {args.input_emb_path}...")
  print("")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--input_emb_path',
                      type=str,
                      default=None,
                      help='path of input emb')
  parser.add_argument(
    '--emb_format',
    type=str,
    default='fasttext',
    help=
    'format of emb, fasttext or word2vec (for word2vec, bin or txt will be inferred from file name)'
  )
  args = parser.parse_args()
  emb_eval(args)
