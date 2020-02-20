# Rethinking Kernel Methods for Node Representation Learning on Graphs

Training code for the paper
**[Rethinking Kernel Methods for Node Representation Learning on Graphs]
(https://arxiv.org/pdf/1910.02548.pdf)**, NIPS 2019

## Overview
Graph kernels are kernel methods measuring graph similarity and serve as a standard tool for graph classification. However, the use of kernel methods for node classification, which is a related problem to graph representation learning, is still ill-posed and the state-of-the-art methods are heavily based on heuristics. Here, we present a novel theoretical kernel-based framework for node classification that can bridge the gap between these two representation learning problems on graphs. Our approach is motivated by graph kernel methodology but extended to learn the node representations capturing the structural information in a graph. We theoretically show that our formulation is as powerful as any positive semidefinite kernels. To efficiently learn the kernel, we propose a novel mechanism for node feature aggregation and a data-driven similarity metric employed during the training phase. More importantly, our framework is flexible and complementary to other graph-based deep learning models, e.g., Graph Convolutional Networks (GCNs). We empirically evaluate our approach on a number of standard node classification benchmarks, and demonstrate that our model sets the new state of the art.

### Prerequisites

This package has the following requirements:

* `Python 3.6`
* `Pytorch 0.4.1`
* `numpy`
* `scipy`
* `networkx`

## Training

python train.py

## Citation

@inproceedings{tian2019rethinking,
  title={Rethinking kernel methods for node representation learning on graphs},
  author={Tian, Yu and Zhao, Long and Peng, Xi and Metaxas, Dimitris},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11681--11692},
  year={2019}
}
