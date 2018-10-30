# TN test

The code in this repository accompanies the experiments performed in the paper [Towards a post-clustering test for differential expression](https://www.biorxiv.org) by Zhang, Kamath, and Tse. 

We provide the following notebooks:
- [seurat_pbmc.ipynb](https://github.com/jessemzhang/tn_test/blob/master/seurat_pbmc.ipynb): R notebook for loading the PBMC and clustering it with Seurat. Please see the Seurat [PBMC tutorial](https://satijalab.org/seurat/pbmc3k_tutorial.html) for more information
- [experiments_pbmc3k.ipynb](https://github.com/jessemzhang/tn_test/blob/master/experiments_pbmc3k.ipynb): Python 3 notebook with TN test experiments performed on PBMC data processed by [seurat_pbmc.ipynb](https://github.com/jessemzhang/tn_test/blob/master/seurat_pbmc.ipynb)
- [experiments_synthetic_normal.ipynb](experiments_synthetic_normal.ipynb): Python 3 notebook with TN test experiments performed on synthetic data
- [figure_utils.py](https://github.com/jessemzhang/tn_test/blob/master/figure_utils.py): Python 3 notebook for preparing the results from the other notebooks for presentation in the manuscript

We also provide two Python modules:
- [truncated_normal.py](https://github.com/jessemzhang/tn_test/blob/master/truncated_normal.py): contains all code required to run the TN test
- [figure_utils.py](https://github.com/jessemzhang/tn_test/blob/master/figure_utils.py): contains code used for running simulations and generating plots

For a tutorial on using the Python modules for your own projects, please refer to [experiments_pbmc3k.ipynb](https://github.com/jessemzhang/tn_test/blob/master/experiments_pbmc3k.ipynb) and [experiments_synthetic_normal.ipynb](experiments_synthetic_normal.ipynb). We were able to run all of our experiments in [this Docker image](https://hub.docker.com/r/heatonresearch/jupyter-python-r/).

## Method

![method](https://github.com/jessemzhang/tn_test/blob/master/method.png)

