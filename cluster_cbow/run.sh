set -x

lang=de
input="../data/$lang/train.txt"
cluster="../word_clusters/$lang.cluster.600"

# minCount is set to 5 as this is stronger baseline than minCount=1.

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
./fasttext cbow -input $input -minCount 5 -dim 200 \
    -output "$lang.cluster.output.emb"  -cluster $cluster \
    -epoch 5 -maxn 0 -neg 5 -loss ns  \
    -freq_thre_in_wd 1 -freq_thre_in_cl 1 -freq_thre_out 100
