# Cluster-incorprated Continuous Bag of Words (CBOW) Model
Implementation of AACL 2020 paper "[A Simple and Effective Usage of Word Clusters for CBOW Model](https://aclanthology.org/2020.aacl-main.10.pdf)"

## Requirements
For using cluster_cbow, which is modified from [fasttext](https://github.com/facebookresearch/fastText), environment that can complie c++ code is required. 

For language modeling task (`word_lm` directory): 
- Python version >= 3.5
- Pytorch version 0.4.0

## Installing clustercat
The word clusters used in this paper are obtained by running [clustercat](https://github.com/jonsafari/clustercat) software on training data. To install `clustercat`, run with
```
git clone https://github.com/jonsafari/clustercat.git
cd clustercat
make
```

## Training word clusters
After installation, run following command to train word clusters over specified corpus:
```
bin/clustercat --min-count 1 -j 15 --classes $class_num < $input_path > $output_path
```
We have uploaded word cluster file under `word_clusters` directory. The format is from the output of clustercat, where the first column is word and the second column is cluster ID separated by tab. In our paper, the clusters are trained only on training data of specific tasks and the number of clusters is 600.

## Compiling cluster_cbow
To compile `cluster_cbow`, run with:
```
cd cluster_cbow
make
```
After this, a binary file `fasttext` will be produced.

## Usage of cluster_cbow
`cluster_cbow` is based on [fasttext](https://github.com/facebookresearch/fastText). For full options, see `./fasttext cbow -h`. The additional options for `cluster_cbow` are as follows:
```
  -cluster            cluster file path
  -freq_thre_in_wd    input words of CBOW whose freq is less than this are not used.
  -freq_thre_in_cl    if freq of input words of CBOW is less than this, their clusters are used. 
  -freq_thre_out      if freq of output words of CBOW is less than this, their clusters are used.
```

## Training word embeddings

The following configuration is used in the paper.

Train cbow without using word clusters:
```
./fasttext cbow -input $input -minCount 5 -dim 200 \
    -output output_emb -epoch 5 -maxn 0 -neg 5 -loss ns
```

Train cbow with word clusters for both input and output:
```
./fasttext cbow -input $input -minCount 5 -dim 200 \
    -output output_emb  -cluster $cluster_path \
    -epoch 5 -maxn 0 -neg 5 -loss ns  \
    -freq_thre_in_wd 100 -freq_thre_in_cl 100 -freq_thre_out 100
```

Train cbow with word cluster only for input:
```
./fasttext cbow -input $input -minCount 5 -dim 200 \
    -output output_emb  -cluster $cluster_path \
    -epoch 5 -maxn 0 -neg 5 -loss ns  \
    -freq_thre_in_wd 100 -freq_thre_in_cl 100 -freq_thre_out 1
```

Train cbow with word cluster only for output:
```
./fasttext cbow -input $input -minCount 5 -dim 200 \
   -output output_emb  -cluster $cluster_path \
   -epoch 5 -maxn 0 -neg 5 -loss ns  \
   -freq_thre_in_wd 1 -freq_thre_in_cl 1 -freq_thre_out 100
```

### Notes
In above code, `minCount=5` is set as this is a stronger baseline than `minCount=1` because many rare words are filtered.
`-maxn` is set to 0 to ensure no subword information is used in fasttext (this paper focused on standard CBOW).
we still observed improvements for CBOW with subword information when word clusters used in output.

## Evaluation on LM

### Dataset
We used a subset of [lmmrl](http://people.ds.cam.ac.uk/dsg40/lmmrl.html) datasets containing 50 different languages ([Gerz et al., 2018](https://www.aclweb.org/anthology/Q18-1032.pdf)). Currently the download link seems broken and I have uploaded one English and German dataset for testing under 'data' directory.

### Run experiments
For evaluating the learned word embedding incorporated with word clusters, first go into `word_lm` directory.

To train a standard neural language model with random initialized word embeddings ('Random' column in Table 6 in paper):
```
python -u main.py --tied  --data $data_path --epoch 40 --emsize 200
```

To train a language model with word embeddings trained from standard CBOW without cluster ('CBOW' column in Table 6 in paper):
```
python -u main.py --input_emb_path cbow_emb_path --tied  --data $data_path --epoch 40 --emsize 200
```

To train a language model with word embeddings trained from cluster-incorporated CBOW ('Cluster-CBOW' column in Table 6 in paper):
```
python -u main.py --input_emb_path cluster_cbow_in_out_emb_path --tied  --data $data_path --epoch 40 --emsize 200
```

To train a language model with word embeddings trained from cluster-incorporated CBOW only for input (Table 8 in paper):
```
python -u main.py --input_emb_path cluster_cbow_in_emb_path --tied  --data $data_path --epoch 40 --emsize 200
```

To train a language model with word embeddings trained from cluster-incorporated CBOW only for output (Table 8 in paper):
```
python -u main.py --input_emb_path cluster_cbow_out_emb_path --tied  --data $data_path --epoch 40 --emsize 200
```
