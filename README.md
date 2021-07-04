# MGP-VAE
Modality Completion via Gaussian Process Prior Variational Autoencoders for Multi-Modal Glioma Segmentation

![](https://github.com/hamghalam/MGP-VAE/blob/main/overview_v3.png)




## Install dependencies

The dependencies can be installed using [anaconda](https://www.anaconda.com/download/):

```bash
conda create -n mgp-vae python=3.6
source activate mgp-vae
conda install -y numpy scipy h5py matplotlib dask pandas
conda install -y pytorch=0.4.1 -c soumit
conda install -y torchvision=0.2.1
```
