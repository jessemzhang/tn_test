# TN test

The code in this repository accompanies the experiments performed in the paper [Towards a post-clustering test for differential expression](https://www.biorxiv.org/content/early/2018/11/05/463265) by Zhang, Kamath, and Tse. 

## Installation

The TN test package can be installed via pip:

```
pip install truncated_normal
```

Import the package by adding the following line of code to your Python script:

```
from truncated_normal import truncated_normal as tn
```

## Examples

For a tutorial on using the TN test module and framework for your own projects, please refer to [tntest_tutorial.ipynb](https://github.com/jessemzhang/tn_test/blob/master/tntest_tutorial.ipynb). We were able to install all required R and Python packages and run all of our experiments in [this Docker image](https://hub.docker.com/r/heatonresearch/jupyter-python-r/).

We also provide the following notebooks for reproducing results in the paper ([figure_utils.py](https://github.com/jessemzhang/tn_test/blob/master/experiments/figure_utils.py) contains code used for running simulations and generating plots):
- [seurat_pbmc.ipynb](https://github.com/jessemzhang/tn_test/blob/master/seurat_pbmc.ipynb): R notebook for loading the PBMC dataset and clustering it with Seurat. Please see the Seurat [PBMC tutorial](https://satijalab.org/seurat/pbmc3k_tutorial.html) for more information
- [experiments_synthetic_normal.ipynb](https://github.com/jessemzhang/tn_test/blob/master/experiments/experiments_synthetic_normal.ipynb): Python 3 notebook with TN test experiments performed on synthetic data
- [experiments_pbmc3k.ipynb](https://github.com/jessemzhang/tn_test/blob/master/experiments/experiments_pbmc3k.ipynb): Python 3 notebook with TN test experiments performed on PBMC data processed by [seurat_pbmc.ipynb](https://github.com/jessemzhang/tn_test/blob/master/seurat_pbmc.ipynb)
- [experiments_kolodziejczyk.ipynb](https://github.com/jessemzhang/tn_test/blob/master/experiments/experiments_kolodziejczyk.ipynb): Python 3 notebook with TN test experiments performed on the mESC dataset published by Kolodzieczyk et al. ([paper](http://www.sciencedirect.com/science/article/pii/S193459091500418X?via%3Dihub), [data](https://github.com/BatzoglouLabSU/SIMLR/tree/SIMLR/data))
- [experiments_zeisel.ipynb](https://github.com/jessemzhang/tn_test/blob/master/experiments/experiments_zeisel.ipynb): Python 3 notebook with TN test experiments performed on the mouse brain cell dataset published by Zeisel et al. ([paper](http://science.sciencemag.org/content/347/6226/1138.full), [data](http://linnarssonlab.org/cortex/))
- Please see the [linear_separability](https://github.com/jessemzhang/tn_test/tree/master/experiments/linear_separability) directory for experiments showing that several published single-cell datasets are linearly separable


## Method

![method](https://github.com/jessemzhang/tn_test/blob/master/method.png)

