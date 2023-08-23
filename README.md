# IGGAN

Unoffical code for the research paper [Inverse Graphics GAN](https://arxiv.org/pdf/2002.12674.pdf).

Under construction.

## RenderNet

Includes a PyTorch implementation of [RenderNet](https://github.com/thunguyenphuoc/RenderNet) as a necessary part of the architecture.

https://github.com/brian-xu/IGGAN/assets/47406953/9b0e1295-ccbb-428a-a977-61247a044d5c

## Notes

CUDA training requires about 10G of VRAM.

## Credits

Generator architecture based on [3DGAN-PyTorch](https://github.com/rimchang/3DGAN-Pytorch).

RenderNet architecture is original code referencing the TensorFlow implementation of
[RenderNet](https://github.com/thunguyenphuoc/RenderNet), since it seems to conflict heavily with the architecture as
described in the paper.

SpectralNorm code taken from
[pytorch-spectral-normalization-gan](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan).

Discriminator architecture based on the
[PyTorch example DCGAN](https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py).

Binvox handling taken from [binvox-rw-py](https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py).
