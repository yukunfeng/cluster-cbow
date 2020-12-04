import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), add_to_vocab=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def id_to_words(self, idx_list):
        word_list = []
        for idx in idx_list:
            word = self.dictionary.idx2word[idx]
            word_list.append(word)
        return word_list

    def tokenize(self, path, add_to_vocab=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                if add_to_vocab:
                    for word in words:
                        self.dictionary.add_word(word)
        unk_tag = "<unk>"
        if add_to_vocab:
            if unk_tag not in self.dictionary.word2idx:
                self.dictionary.add_word(unk_tag)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word not in self.dictionary.word2idx:
                        word = unk_tag
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


if __name__ == "__main__":
   corpus = Corpus("./tmp/") 
   train = corpus.id_to_words(corpus.train)
   print(f"train: {train}")
   valid = corpus.id_to_words(corpus.valid)
   print(f"valid: {valid}")
   test = corpus.id_to_words(corpus.test)
   print(f"test: {test}")

