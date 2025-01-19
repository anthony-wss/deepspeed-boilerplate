# DeepSpeed Boilerplate

## Train 1B model

1. Build the singularity image with >50GB RAM job. This can take ~30min.
- On battleship, you may have to manually copy `/usr/bin/mksquashfs` to your `$PATH` since it's not accessable on compute node.

```bash
singularity pull docker://hsiuhsuan/deepspeed
```

2. Clone this repo
```bash
git clone https://github.com/anthony-wss/deepspeed-boilerplate.git
```

3. Run the training
```bash
srun --gpus-per-node 2 --account XXX -N 1 -n 1 --pty /bin/bash
singularity shell --nv deepspeed_latest.sif
source ~/.bashrc && conda activate deepspeed
cd 1b
deepspeed python train.py
```

## Experiments

### 1b model

| nGPUs | estimated GPU VRAM | measured per GPU VRAM |
|-------|----------|----------|
| 1 | 21.70GB | 35GB   |
| 2 | 11.34GB | 23GB   |
| 4 | 6.16GB | 15.8GB |
| 8 | 3.57GB | 10.5GB |

## Common Issues

1. The server socket has failed to listen on any local network address

```
  File "/home/u3937558/.local/lib/python3.10/site-packages/torch/distributed/rendezvous.py", line 185, in _create_c10d_store
    return TCPStore(
RuntimeError: The server socket has failed to listen on any local network address. useIpv6: 0, code: -98, name: EADDRINUSE, message: address already in use
```

Solution: add `--master_port xxxxx` to select a free port

