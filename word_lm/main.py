# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torchtext

import data
import model
from word_emb import load_word_emb

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='cuda')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--input_emb_path', type=str, default=None,
                    help='path of input emb')
parser.add_argument('--emb_format', type=str, default='fasttext',
                    help='format of emb, fasttext or word2vec (for word2vec, bin or txt will be inferred from file name)')
parser.add_argument('--verbose_test_file', type=str, default=None,
                    help='path of output emb')
args = parser.parse_args()
print(args)
if args.tied:
    args.nhid = args.emsize

# temp codes
#  tmp = os.path.expanduser(args.data)
#  head, tail = os.path.split(tmp)
#  _, tail = os.path.split(head)
#  if tail in ["en", "ja", "zh", "ar", "am"]:
    #  print(f"skip {tail}")
    #  exit(0)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    pass
    #  if not args.cuda:
        #  print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#  device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device(args.device)
gpu_id = int(args.device.split(':')[1])
torch.cuda.set_device(gpu_id)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
print(f"vocab: {len(corpus.dictionary.idx2word)}")

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

##########################
#  load pre-trained emb  #
##########################

# This code just loads word embs but is redundant due to historical reasons.
#  if args.input_emb_path is not None and args.input_emb_path != "None":
    #  print("start to load outemb")
    #  input_emb_path = os.path.expanduser(args.input_emb_path)
    #  os.system(f"rm -f {input_emb_path}.pt")
    #  head, tail = os.path.split(input_emb_path)
    #  torchtext_vectors = torchtext.vocab.Vectors(name=tail, cache=head)
    #  torchtext_vectors.vectors.to(device)
    #  ttword_index_list = []
    #  for i in range(len(corpus.dictionary)):
        #  word = corpus.dictionary.idx2word[i]
        #  if word not in torchtext_vectors.stoi:
            #  word = "</s>"
        #  torchtext_vec_index = torchtext_vectors.stoi[word]
        #  ttword_index_list.append(torchtext_vec_index)
    #  ttword_index_list = torch.tensor(ttword_index_list)
    #  outemb = torch.index_select(
        #  torchtext_vectors.vectors, 0, ttword_index_list
    #  )
    ##  turn back unk to random vectors if it's not in pretrained emb
    #  if "<unk>" not in torchtext_vectors.stoi:
        #  unk_idx = corpus.dictionary.word2idx["<unk>"]
        ##  outemb[unk_idx].zero_()
        #  outemb[unk_idx].uniform_(-0.1, 0.1)

    #  os.system(f"rm -f {input_emb_path}.pt")
    #  print(f"finish to load outemb, size: {outemb.size()}")


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

# Load emb
if args.input_emb_path is not None and args.input_emb_path != "None":
    input_emb_path = os.path.expanduser(args.input_emb_path)
    emb_model = load_word_emb(input_emb_path, emb_format)
    found = 0
    for i in range(len(corpus.dictionary)):
        word = corpus.dictionary.idx2word[i]
        if word in emb_model:
            found += 1
            word_emb = torch.from_numpy(emb_model[word])
            # Assign to model's lookup table
            model.encoder.weight.data[i] = word_emb
            if not args.tied:
                model.decoder.weight.data[i] = word_emb
    found_rate = found / len(corpus.dictionary)
    print("Finishing to load emb from {input_emb_path}, {found}/{len(corpus.dictionary)}={found_rate * 100:.2f}")
    import ipdb; ipdb.set_trace()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def verbose_test(idx2word, counter, word_indexs, scores):
    print_contents = []
    freq_loss = 0
    freq_count = 0
    infreq_loss = 0
    infreq_count = 0
    for row_i, row in enumerate(word_indexs, 0):
        line_out = ""
        for col_i, word_index in enumerate(row, 0):
            word = idx2word[word_index]
            score = scores[row_i][col_i]
            freq = f"unk-{word}"
            if word in counter:
                freq = counter[word]
                if freq >= 100:
                    freq_loss += score
                    freq_count += 1
                else:
                    infreq_loss += score
                    infreq_count += 1
            line_out += f"{word}[score:{score:5.2f},freq:{freq}]  "
        line_score = scores[row_i].sum()
        line_out = f"{line_score:5.1f}: {line_out}"
        print_contents.append(line_out)
    return print_contents, [freq_loss, freq_count], [infreq_loss, infreq_count]


def evaluate(data_source, verbose=False):
    # Turn on evaluation mode which disables dropout.
    if verbose:
        from collections import Counter
        counter = Counter()
        train_file = f"{args.data}/train.txt"
        lines = [
            counter.update(line.strip().split()) for line in open(train_file, 'r').readlines()
        ]
        #  fh_out = open(args.verbose_test_file, "w")
        verbose_criterion = nn.CrossEntropyLoss(reduce=False)

    model.eval()
    total_loss = 0.
    total_freq_loss = 0.
    total_freq_count = 0
    total_infreq_loss = 0.
    total_infreq_count = 0

    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
            if verbose:
                verbose_loss = verbose_criterion(output_flat, targets)
                verbose_loss = verbose_loss.view(data.size(0), -1)
                print_contents, [freq_loss, freq_count], [infreq_loss, infreq_count] = verbose_test(
                    corpus.dictionary.idx2word, counter,
                    data.t(), verbose_loss.t()
                )
                total_freq_loss += freq_loss
                total_freq_count += freq_count
                total_infreq_loss += infreq_loss
                total_infreq_count += infreq_count
                #  for print_line in print_contents:
                    #  fh_out.write(f"{print_line}\n")

    if verbose:
        #  fh_out.close()
        return math.exp(total_freq_loss / total_freq_count), math.exp(total_infreq_loss / total_infreq_count)
    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            #  print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    #  'loss {:5.2f} | ppl {:8.2f}'.format(
                #  epoch, batch, len(train_data) // args.bptt, lr,
                #  elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        #  print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | lr {:8.3f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss), lr))
        #  print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            #  with open(args.save, 'wb') as f:
                #  torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
        if epoch % 4 == 0:
            test_loss = evaluate(test_data)
            print(f"test ppl : {math.exp(test_loss)}")
        if lr < 0.05:
            break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
#  with open(args.save, 'rb') as f:
    #  model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    #  model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | best_val_ppl {:8.2f} test ppl {:8.2f}'.format(
    test_loss, math.exp(best_val_loss), math.exp(test_loss)))
print('=' * 89)
print(f"{args.data}--{args.input_emb_path} {math.exp(best_val_loss)} {math.exp(test_loss)}")
# run logging files
#  if args.verbose_test_file:
freq_ppl, infreq_ppl = evaluate(val_data, verbose=True)
print(f"freq ppl: {freq_ppl}, infreq ppl: {infreq_ppl}")

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
