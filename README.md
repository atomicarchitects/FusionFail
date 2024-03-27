# Fusion Fail

Profile showing the break down of kernel fusion for one layer of NequIP. Thanks @ameya98 for the simplified version of NequIP !



### Packages

```bash
pip install requirements.txt
```

To reproduce the profile shown above install Nsight Systems and run

```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nequip_profile -f true --cudabacktrace=true -x true python train.py
```