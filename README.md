# Rethinking Kernel Methods for Node Representation Learning on Graphs

Training code for the paper
**[Rethinking Kernel Methods for Node Representation Learning on Graphs]
(https://arxiv.org/pdf/1910.02548.pdf)**, NIPS 2019

## Overview
We present a novel theoretical kernel-based framework for node classification. Our approach is motivated by graph kernel methodology but extended to learn the node representations capturing the structural information in a graph. We theoretically show that our formulation is as powerful as any positive semidefinite kernels. Our framework is flexible and complementary to other graph-based deep learning models, e.g., Graph Convolutional Networks (GCNs).
<p align="center"><img src="nips19_poster.png" alt="poster" width="1000"></p>

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
If you find this code useful in your research, please consider citing:
```
@inproceedings{tian2019rethinking,
  title={Rethinking kernel methods for node representation learning on graphs},
  author={Tian, Yu and Zhao, Long and Peng, Xi and Metaxas, Dimitris},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11681--11692},
  year={2019}
}
```
