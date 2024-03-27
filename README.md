# Fusion Fail

Profile showing the break down of kernel fusion for one layer of NequIP. Thanks @ameya98 for the simplified version of NequIP !

![nequip_profile](profiles/profile_train_step_nequip_3_layers.png)


## What are we looking at ?

The function `train_step` corresponds to a forward and backward pass through a 3 layered NequIP model implemented using e3nn-jax acting on a simple Tetris dataset.


### Packages

```bash
pip install requirements.txt
```

To reproduce the profile shown above install Nsight Systems and run

```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nequip_profile -f true --cudabacktrace=true -x true python train.py
```