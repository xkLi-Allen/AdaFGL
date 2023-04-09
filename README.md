## AdaFGL: A New Paradigm for Federated Graph Learning

**Requirements**

Hardware environment: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz, NVIDIA GeForce RTX 3090 with 24GB memory.

Software environment: Ubuntu 18.04.6, Python 3.9, PyTorch 1.11.0 and CUDA 11.8.

1. Please refer to [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install the environments;
2. Run 'pip install -r requirements.txt' to download required packages;

**Training**

To train the model(s) in the paper

1. Please unzip Planetoid.zip/Chameleon.zip to the current file directory location
2. Open main.py to train AdaFGL with our proposed decouple two step training framework.

    We provide Planetoid/Chameleon dataset under Louvain and structure Noniid 10 clients split as example (hyperparameters in our paper).

    Run this command:

```python
  python main.py
```
