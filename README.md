# Graph Constrained Data Representation Learning for Human Motion Segmentation

The repository contains our results and PyTorch code for the proposed algorithm described in our Graph Constrained Data Representation Learning for Human Motion Segmentation ICCV 2021 [paper](https://arxiv.org/pdf/2107.13362.pdf).

## <sub> Abstract </sub>
Recently, transfer subspace learning based approaches have shown to be a valid alternative to unsupervised subspace clustering and temporal data clustering  for human motion segmentation (HMS). These approaches leverage prior knowledge from a source domain to improve clustering performance on a target domain, and currently they represent the state of the art in HMS.

Bucking this trend, in this paper, we propose a novel unsupervised model that learns a representation of the data and digs clustering information from the data itself. Our model is reminiscent of temporal subspace clustering, but presents two critical differences. 
First, we learn an auxiliary data matrix that can deviate from the initial data, hence confers more degrees of freedom to the coding matrix. 
Second, we introduce a regularization term for this auxiliary data matrix that preserves the local geometrical structure present in the high-dimensional space.
The proposed model is efficiently optimized by using an original Alternating Direction Method of Multipliers (ADMM) formulation allowing to learn jointly the auxiliary data representation, a nonnegative dictionary and a coding matrix. 

Experimental results on four benchmark datasets for HMS demonstrate that our approach achieves significantly better clustering performance then state-of-the-art methods, including both unsupervised and more recent semi-supervised transfer learning approaches. 

#
![ImagenDGE-TSC-NoNLSS-2-v1](https://user-images.githubusercontent.com/50593288/129836762-22641599-dc30-415d-a74c-0f6dabc665cc.png)

## Citation

```js
@InProceedings{Dimiccoli_2021_ICCV,
author = {Dimiccoli, Mariella and Garrido, Lluis and Rodriguez-Corominas, Guillem and Wendt, Herwig},
title = {Graph Constrained Data Representation Learning for Human Motion Segmentation},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021}
}
```
