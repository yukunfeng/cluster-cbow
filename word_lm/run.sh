set -x

log="log"

lang=de
data_path="../data/$lang"

# word-lm with random initialized word embeddings.
# 'Random' column in Table 6 in paper.
# python -u main.py --tied  --data $data_path --epoch 40 --emsize 200  >> $log

# word-lm with word embeddings trained from standard CBOW without cluster.
# 'CBOW' column in Table 6 in paper.
# python -u main.py --input_emb_path "../cluster_cbow/$lang.no.cluster.emb" --tied  --data $data_path --epoch 40 --emsize 200  >> $log

# word-lm with word embeddings trained from cluster-incorporated CBOW.
# 'Cluster-CBOW' column in Table 6 in paper.
# python -u main.py --input_emb_path "../cluster_cbow/$lang.cluster.emb" --tied  --data $data_path --epoch 40 --emsize 200  >> $log

# word-lm with word embeddings trained from cluster-incorporated CBOW only for input.
# Table 8 in paper.
# python -u main.py --input_emb_path "../cluster_cbow/$lang.cluster.input.emb" --tied  --data $data_path --epoch 40 --emsize 200  >> $log

# word-lm with word embeddings trained from cluster-incorporated CBOW only for output.
# Table 8 in paper.
# python -u main.py --input_emb_path "../cluster_cbow/$lang.cluster.output.emb" --tied  --data $data_path --epoch 40 --emsize 200  >> $log
