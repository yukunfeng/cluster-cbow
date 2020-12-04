# Cluster-incorprated Continuous Bag of Words (CBOW) Model
Implementation of AACL 2020 paper "[A Simple and Effective Usage of Word Clusters for CBOW Model](https://www.aclweb.org/toadd)"

## Requirements
For using cluster_cbow, which is modified from [fasttext](https://github.com/facebookresearch/fastText), environment that can complie c++ code is required. 

For language modeling task (`word_lm` directory): 
- Python version >= 3.5
- Pytorch version 0.4.0

## 1 Step
The word clusters used in this paper are obtained by running [clustercat](https://github.com/jonsafari/clustercat) software on training data. We have uploaded one under `word_clusters` directory. The format is from the output of clustercat, where the first column is word and the second column is cluster ID separated by tab. 
##

## 2 Step
Go into `cluster_cbow` directory and compile with `make` command.
This will produce executable file. For example and hyper-parameters explanation, see comments in `run.sh` under `cluster_cbow`.

## 3 Step
For evaluating the learned word embedding incorporated with word clusters, go into `word_lm` directory, see comments in `run.sh` for details.
