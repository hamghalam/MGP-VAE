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

## Preprocessing

The brain dataset is available at [?].
The data can be preprocessed as follows:

```bash
cd MGPVAE/pysrc/brain
python process_data_mri.py
```

## Run VAE

Plots and weights are saved in the "/out/vae".

```bash
python train_vae_mri.py --outdir ./out/vae
```

## Run MGPVAE
The autoencoder parameters of the MGPVAE model should be initialized to the last step pre-trained VAE for the best performance. 

For instance, if the "vae" results are held in "./out/vae" and one has trained VAE for 10000 epochs, then we can use:

```bash
python train_gppvae_mri.py --outdir ./out/gppvae --vae_cfg ./out/vae/vae.cfg.p --vae_weights ./out/vae/weights.09900.pt
```





## GPPVAE
we extend the GPPVAE for missing MRI sub-modalities imputation in a 3D framework for brain glioma tumor segmentation.
```
@article{casale2018gaussian,
  title={Gaussian Process Prior Variational Autoencoders},
  author={Casale, Francesco Paolo and Dalca, Adrian V and Saglietti, Luca and Listgarten, Jennifer and Fusi, Nicolo},
  journal={32nd Conference on Neural Information Processing Systems},
  year={2018}
}
```

## Citation

If you use any part of this code in your research, please cite our [paper](?):

```
@article{hamghalam2021,
  title={Modality Completion via Gaussian Process Prior Variational Autoencoders for Multi-Modal Glioma Segmentation},
  author={Hamghalam, Mohammad and Frangi, Alejandro F.  and Lei, Baiying and Simpson, Amber L.},
  journal={the 24th International Conference on Medical Image Computing and Computer Assisted Intervention},
  year={2021}
}
```
