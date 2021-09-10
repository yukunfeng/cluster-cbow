set -x

log="log"

data_path="/home/lr/yukun/common_corpus/data/50lm/en"

# word-lm with random initialized word embeddings.
# Random word embeddings
# python -u main.py --tied  --data $data_path --epoch 40 --emsize 300  >> $log

# word-lm with word embeddings trained from standard subword-CBOW without cluster.
# cluster=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.fasttext.emb
# python -u main.py --input_emb_path $cluster --tied  --data $data_path --epoch 40 --emsize 300  >> $log

# word-lm with word embeddings trained from standard subword-CBOW without cluster.
cluster=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.emb
python -u main.py --input_emb_path $cluster --tied  --data $data_path --epoch 40 --emsize 300  >> $log

# word-lm with word embeddings trained from cluster-incorporated CBOW only for output.
# cluster=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.outfreq100.emb
# python -u main.py --input_emb_path $cluster --tied  --data $data_path --epoch 40 --emsize 300  >> $log

# word-lm with word embeddings trained from cluster-incorporated CBOW only for output.
# cluster=/home/lr/yukun/large_scale_corpus/preprocessed_wiki.txt.no_rare.txt.shuf.fasttext.outfreq50.emb
# python -u main.py --input_emb_path $cluster --tied  --data $data_path --epoch 40 --emsize 300  >> $log
