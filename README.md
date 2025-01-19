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

### 8b model

#### 1 gpu

```
  per CPU  |  per GPU |   Options
  201.93GB |   1.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
  201.93GB |   1.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
  179.49GB |  16.91GB | offload_param=none, offload_optimizer=cpu , zero_init=1
  179.49GB |  16.91GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    2.94GB | 136.57GB | offload_param=none, offload_optimizer=none, zero_init=1
   44.87GB | 136.57GB | offload_param=none, offload_optimizer=none, zero_init=0
```

#### 2 gpu

```
  per CPU  |  per GPU |   Options
  201.93GB |   1.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
  201.93GB |   1.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
  179.49GB |   9.44GB | offload_param=none, offload_optimizer=cpu , zero_init=1
  179.49GB |   9.44GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    5.87GB |  69.27GB | offload_param=none, offload_optimizer=none, zero_init=1
   89.75GB |  69.27GB | offload_param=none, offload_optimizer=none, zero_init=0
```

#### 4 gpu

```
  per CPU  |  per GPU |   Options
  201.93GB |   1.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
  201.93GB |   1.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
  179.49GB |   5.70GB | offload_param=none, offload_optimizer=cpu , zero_init=1
  179.49GB |   5.70GB | offload_param=none, offload_optimizer=cpu , zero_init=0
   11.74GB |  35.61GB | offload_param=none, offload_optimizer=none, zero_init=1
  179.49GB |  35.61GB | offload_param=none, offload_optimizer=none, zero_init=0
```

Measured:
- offload optimizer: 15GB per gpu, 74s / step (batch_size=1, max_length=512)
    - Set `OMP_NUM_THREADS=4` does not help

#### 8 gpu

```
  per CPU  |  per GPU |   Options
  201.93GB |   1.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
  358.98GB |   1.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
  179.49GB |   3.83GB | offload_param=none, offload_optimizer=cpu , zero_init=1
  358.98GB |   3.83GB | offload_param=none, offload_optimizer=cpu , zero_init=0
   23.48GB |  18.78GB | offload_param=none, offload_optimizer=none, zero_init=1
  358.98GB |  18.78GB | offload_param=none, offload_optimizer=none, zero_init=0
```

Measured:
- no offload: 24-32GB(varies but will not OOM) per gpu, 7.3s / step (batch_size=1, max_length=512)

## Common Issues

1. The server socket has failed to listen on any local network address

```
  File "/home/u3937558/.local/lib/python3.10/site-packages/torch/distributed/rendezvous.py", line 185, in _create_c10d_store
    return TCPStore(
RuntimeError: The server socket has failed to listen on any local network address. useIpv6: 0, code: -98, name: EADDRINUSE, message: address already in use
```

Solution: add `--master_port xxxxx` to select a free port

