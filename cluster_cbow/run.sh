
# -minCount is set to 5 as this is stronger baseline than minCount=1.
# The reason why word embeddings trained from standard CBOW can improve shown in this paper is because we set minCount.
# Rare words will be disgarded and thus will benefit the quality of other word embeddings.

# -maxn is set to 0 to ensure no subword information is used in fasttext (this paper focused on standard CBOW).
# We tried to combine word cluster and subwords in input but there is no obvious improvement. 
# However, we observed word clusters used in output in CBOW can benefit when subword information is used. The reason has been mentioned in paper.

# -freq_thre_in_wd n: if the frequency of one word <= n, this word will not be used in input of CBOW.
# -freq_thre_in_cl n: if the frequency of one word <= n, we use its cluster instead in input of CBOW.
# -freq_thre_out n. : if the frequency of one word <= n, we use its cluster instead in output of CBOW.

# other hyper-parameters from fasttext, see ./fasttext -h.

set -x

lang=de
input="../data/$lang/train.txt"
cluster="../word_clusters/$lang.cluster.600"

# cbow without using word clusters.
# ./fasttext cbow -input $input -minCount 5 -dim 200 \
    # -output "$lang.no.cluster.emb" -epoch 5 -maxn 0 -neg 5 -loss ns

# cbow with word clusters for both input and output.
# ./fasttext cbow -input $input -minCount 5 -dim 200 \
    # -output "$lang.cluster.emb"  -cluster $cluster \
    # -epoch 5 -maxn 0 -neg 5 -loss ns  \
    # -freq_thre_in_wd 100 -freq_thre_in_cl 100 -freq_thre_out 100

# cbow with word cluster only for input
# ./fasttext cbow -input $input -minCount 5 -dim 200 \
    # -output "$lang.cluster.input.emb"  -cluster $cluster \
    # -epoch 5 -maxn 0 -neg 5 -loss ns  \
    # -freq_thre_in_wd 100 -freq_thre_in_cl 100 -freq_thre_out 1

# cbow with word cluster only for output
# ./fasttext cbow -input $input -minCount 5 -dim 200 \
   # -output "$lang.cluster.output.emb"  -cluster $cluster \
   # -epoch 5 -maxn 0 -neg 5 -loss ns  \
   # -freq_thre_in_wd 1 -freq_thre_in_cl 1 -freq_thre_out 100
