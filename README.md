# Neural Graph Generator

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for the paper [Neural Graph Generator: Feature-Conditioned Graph Generation
using Latent Diffusion Models](http://arxiv.org/abs/2403.01535).


### Abstract
Graph generation has emerged as a crucial task in machine learning, with significant challenges in generating graphs that accurately reflect specific properties. Existing methods often fall short in efficiently addressing this need as they struggle with the high-dimensional complexity and varied nature of graph properties. In this paper, we introduce the Neural Graph Generator (NGG), a novel approach which utilizes conditioned latent diffusion models for graph generation. NGG demonstrates a remarkable capacity to model complex graph patterns, offering control over the graph generation process. NGG employs a variational graph autoencoder for graph compression and a diffusion process in the latent vector space, guided by vectors summarizing graph statistics. We demonstrate NGG's versatility across various graph generation tasks, showing its capability to capture desired graph properties and generalize to unseen larger graphs. This work signifies a significant shift in graph generation methodologies, offering a more practical and efficient solution for generating diverse types of graphs with specific characteristics.

### Overview of the proposed architecture

<div align=center>
<img src=https://github.com/iakovosevdaimon/Graph-Generator/blob/main/figures/ldm.jpg width="100%">
</div>


### Requirements
Code is written in Python 3.8 and requires:
* NetworkX 2.6
* python-louvain 0.16
* Torch-geometric 2.2.0
* Pytorch 1.13.1
* GraKeL 0.1.8 

### Dataset
1M synthetic graphs with their properties.
Can be downloaded through this [link](https://drive.google.com/file/d/1NMJiAKwgAxq8l1XYZ4KGxj2NMuuq-TUl/view?usp=sharing).
Soon available on Pytorch Geometric.



### Run the model
First, specify the dataset and the hyperparameters in the `main.py` file. Then, use the following command:

```
$ python main.py
```

Arguments:
```
--lr "Initial Learning rate"
--dropout "Dropout rate"
--batch-size "Input batch size for training"
--epochs-autoencoder "Number of epochs to train autoencoder"
--hidden-dim-encoder "Size of hidden layer of encoder"
--hidden-dim-decoder "Size of hidden layer of decoder"
--latent-dim "Size of latent representation"
--n-max-nodes "Maximum number of nodes for the graphs of our dataset"
--n-layers-encoder "Number of layers of encoder"
--n-layers-decoder "Number of layers of decoder"
--spectral-emb-dim "Size of spectral embeddings"
--variational-autoencoder "Set this argument if you want to use the Variational Autoencoder"
--epochs-denoise "Number of epochs to train diffusion model"
--timesteps "Number of timesteps for the diffusion process"
--hidden-dim-denoise "Size of hidden layer of denoiser"
--n-layers_denoise "Number of layers of denoiser"
--train-autoencoder "Set this argument if you want to train autoencoder"
--train-denoiser "Set this argument if you want to train diffusion model"
--n-properties "Number of graph properties"
--dim-condition "Size of hidden layer of conditioner MLP"

```

### Cite
Please cite our paper if you use this code or our dataset:
```
@misc{evdaimon2024neural,
      title={Neural Graph Generator: Feature-Conditioned Graph Generation using Latent Diffusion Models}, 
      author={Iakovos Evdaimon and Giannis Nikolentzos and Michail Chatzianastasis and Hadi Abdine and Michalis Vazirgiannis},
      year={2024},
      eprint={2403.01535},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
